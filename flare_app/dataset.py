import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def get_page_index(site_name) -> int:
    """Derive page index from the NVFlare site name (e.g. 'site-1' -> 0)."""
    match = re.search(r"(\d+)$", site_name)
    if not match:
        raise ValueError(f"cannot parse page index from site name: {site_name}")
    return int(match.group(1)) - 1


def read_num_classes(manifest_path: Path):
    all_data = read_data(manifest_path)
    label_table = (all_data[["label", "label_index"]].drop_duplicates()
                   .sort_values("label_index"))
    class_names = label_table["label"].astype(str).tolist()
    return len(class_names)


def read_data(manifest_path: Path):
    """Read the full manifest."""
    df = pd.read_csv(manifest_path, encoding="utf-8")

    required = {"sample_id", "image_path", "label", "label_index"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(df.columns))}")

    return df


def split_frame(frame, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Stratified split into train/val/test, keeping label distribution."""
    train, rest = train_test_split(frame, test_size=val_ratio + test_ratio, stratify=frame["label_index"],
                                   random_state=seed)
    val, test = train_test_split(rest, test_size=test_ratio / (val_ratio + test_ratio), stratify=rest["label_index"],
                                 random_state=seed)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def read_page(frame: pd.DataFrame, page_index: int, num_pages: int) -> pd.DataFrame:
    """Slice one page out of a dataframe by row count.

    The first `remainder` pages get one extra row.
    """
    if num_pages < 1:
        raise ValueError(f"num_pages must be >= 1, got {num_pages}")
    if not 0 <= page_index < num_pages:
        raise ValueError(f"page_index must be in [0, {num_pages}), got {page_index}")

    n = len(frame)
    base = n // num_pages
    remainder = n % num_pages

    start = page_index * base + min(page_index, remainder)
    size = base + (1 if page_index < remainder else 0)

    page = frame.iloc[start: start + size].reset_index(drop=True)
    return page


class CurveImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label_index"]), str(row["sample_id"])
