import argparse
import json
import math
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import nvflare.client as flare  # 新增
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, log_loss)

from config import Config
from flare_app.dataset import CurveImageDataset, read_data, split_frame, read_page, get_page_index
from flare_app.model import WaveletConvNeXtTiny


data_root = Path(Config.RUNTIME_DATA_ROOT)
manifest_path = data_root / "processed" / "convnext" / "image_manifest.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=manifest_path)
    p.add_argument("--num_clients", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/convnext"))
    p.add_argument("--pretrained-checkpoint", type=Path, default=None,
                   help="Optional official ConvNeXt-T checkpoint; compatible weights are transferred")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--wavelet-levels", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def load_partial_convnext_weights(model: nn.Module, path: Path) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state = checkpoint
    state = {key.removeprefix("module.").removeprefix("model."): value
             for key, value in state.items()}
    target = model.state_dict()
    accepted = {}
    for key, value in state.items():
        if key in target and target[key].shape == value.shape:
            accepted[key] = value
            continue
        # Original ConvNeXt depthwise 7x7 -> WTConv base branch.
        if key.endswith(".dwconv.weight") or key.endswith(".dwconv.bias"):
            mapped = key.replace(".dwconv.weight", ".dwconv.base_conv.weight").replace(
                ".dwconv.bias", ".dwconv.base_conv.bias"
            )
            if mapped in target and target[mapped].shape == value.shape:
                accepted[mapped] = value
    missing, unexpected = model.load_state_dict(accepted, strict=False)
    return {"loaded": len(accepted), "missing": len(missing), "unexpected": len(unexpected)}


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray,
                      class_names: list[str]) -> dict:
    prediction = probabilities.argmax(axis=1)
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "multiclass_log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, prediction, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0
        ),
    }


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: Optional[torch.optim.Optimizer],
              scaler, amp_enabled: bool, accumulation: int,
              grad_clip: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_probabilities, all_targets, all_ids = [], [], []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (images, target, sample_ids) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            autocast_context = (torch.autocast(device_type="cuda", enabled=True)
                                if amp_enabled else nullcontext())
            with autocast_context:
                logits = model(images)
                loss = criterion(logits, target)
            if training:
                scaler.scale(loss / accumulation).backward()
                should_step = step % accumulation == 0 or step == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * len(target)
        all_probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        all_ids.extend(sample_ids)
    return (total_loss / len(loader.dataset), np.concatenate(all_probabilities),
            np.concatenate(all_targets), np.asarray(all_ids, dtype=str))


def make_transforms(image_size: int):
    normalise = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        # No flips, crops, translations or random scaling: loop position and
        # physical width/height are classification features.
        transforms.ToTensor(),
        normalise,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), normalise,
    ])
    return train_transform, eval_transform


def build_loader(dataset: Dataset, batch_size: int, workers: int,
                 shuffle: bool) -> DataLoader:
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def write_predictions(path: Path, sample_ids: np.ndarray, y_true: np.ndarray,
                      probabilities: np.ndarray, class_names: list[str]) -> None:
    prediction = probabilities.argmax(axis=1)
    data: dict[str, object] = {
        "sample_id": sample_ids,
        "true_index": y_true,
        "true_label": [class_names[i] for i in y_true],
        "pred_index": prediction,
        "pred_label": [class_names[i] for i in prediction],
    }
    for index, name in enumerate(class_names):
        data[f"prob__{index}__{name}"] = probabilities[:, index]
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive.")
    seed_everything(args.seed)
    device = select_device(args.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # FLARE federated learning
    flare.init()

    # read data
    # 1. 读完整 manifest
    all_data = read_data(args.manifest)

    # 2. 先分层切分 train/val/test
    train_data, val_data, test_data = split_frame(all_data)

    # 3. 再按页取自己那一份
    page_index = get_page_index(flare.get_site_name())
    train_data = read_page(train_data, page_index, args.num_clients)
    val_data = read_page(val_data, page_index, args.num_clients)
    test_data = read_page(test_data, page_index, args.num_clients)

    # all_data = read_data(args.manifest, flare.get_site_name(), args.num_clients)
    # train_data, val_data, test_data = split_frame(all_data, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")

    label_table = (all_data[["label", "label_index"]].drop_duplicates()
                   .sort_values("label_index"))
    class_names = label_table["label"].astype(str).tolist()
    # print(f'label_table: {label_table}')
    # print(f'class_names: {class_names}')
    if label_table["label_index"].astype(int).tolist() != list(range(len(class_names))):
        raise ValueError("label_index must be contiguous and start at zero.")
    # frames = {part: manifest.loc[manifest["split"] == part].copy()
    #           for part in ("train", "val", "test")}
    # if any(frame.empty for frame in frames.values()):
    #     raise ValueError("Manifest must contain non-empty train, val and test splits.")

    train_transform, eval_transform = make_transforms(args.image_size)
    root = args.manifest.parent
    datasets = {
        "train": CurveImageDataset(train_data, root, train_transform),
        "val": CurveImageDataset(val_data, root, eval_transform),
        "test": CurveImageDataset(test_data, root, eval_transform),
    }
    loaders = {
        part: build_loader(dataset, args.batch_size, args.workers, shuffle=(part == "train"))
        for part, dataset in datasets.items()
    }

    # model init
    model = WaveletConvNeXtTiny(num_classes=len(class_names), wavelet_levels=args.wavelet_levels, drop_path_rate=args.drop_path)
    transfer = None
    if args.pretrained_checkpoint:
        transfer = load_partial_convnext_weights(model, args.pretrained_checkpoint)
        print(f"Transferred checkpoint tensors: {transfer}")
    model.to(device)
    train_targets = train_data["label_index"].to_numpy(dtype=int)
    counts = np.bincount(train_targets, minlength=len(class_names))
    if np.any(counts == 0):
        raise ValueError(f"Training split misses classes: {np.flatnonzero(counts == 0).tolist()}")
    class_weights = len(train_targets) / (len(class_names) * counts)

    # loss function, optimizer and scheduler ───
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    warmup_epochs = min(5, max(1, args.epochs // 10))
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                               total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=args.learning_rate * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "convnext_wavelet_tiny_best.pt"

    while flare.is_running():
        # 1. receive the global model
        input_model = flare.receive()

        # 2. override local model
        if input_model.params is not None:
            model.load_state_dict(input_model.params)

        # 3. local training
        history, best_f1, stale = [], -math.inf, 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_prob, train_y, _ = run_epoch(
                model, loaders["train"], criterion, device, optimizer, scaler,
                amp_enabled, args.grad_accum_steps, args.grad_clip
            )
            val_loss, val_prob, val_y, _ = run_epoch(
                model, loaders["val"], criterion, device, None, scaler,
                amp_enabled, 1, args.grad_clip
            )
            scheduler.step()
            train_macro = f1_score(train_y, train_prob.argmax(1), average="macro", zero_division=0)
            val_macro = f1_score(val_y, val_prob.argmax(1), average="macro", zero_division=0)
            record = {
                "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss, "train_macro_f1": float(train_macro),
                "val_loss": val_loss, "val_macro_f1": float(val_macro),
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))
            if val_macro > best_f1 + 1e-6:
                best_f1, stale = float(val_macro), 0
                torch.save({
                    "model": model.state_dict(), "class_names": class_names,
                    "epoch": epoch, "val_macro_f1": best_f1,
                    "architecture": "WaveletConvNeXt-T", "wavelet_levels": args.wavelet_levels,
                }, best_path)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping at epoch {epoch}; best validation macro-F1={best_f1:.6f}")
                    break

        # 4. 评估全局模型当前 round 的性能（可选，用于服务器端指标记录）
        _, val_prob, val_y, _ = run_epoch(
            model, loaders["val"], criterion, device, None,
            scaler, amp_enabled, 1, args.grad_clip
        )
        val_metrics = calculate_metrics(val_y, val_prob, class_names)

        # 5. 发送更新后的模型
        output_model = flare.FLModel(
            params=model.state_dict(),
            metrics={"val_macro_f1": val_metrics["macro_f1"],
                     "val_accuracy": val_metrics["accuracy"]},
            meta={"NUM_STEPS_CURRENT_ROUND": len(loaders["train"])},
        )
        flare.send(output_model)

        # 6. 可选：测试集评估（仅在需要时执行，避免每轮耗时）
        # 建议放在 job 结束后单独跑，或仅在最后一轮做


if __name__ == "__main__":
    main()
