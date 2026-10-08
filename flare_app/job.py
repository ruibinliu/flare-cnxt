
import shlex
from pathlib import Path

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv  #, add_experiment_tracking

from config.config import Config
from flare_app.dataset import read_num_classes


def create_recipe():
    train_args = ["--epochs", str(Config.NUM_EPOCHS)]
    train_args.extend(["--batch-size", str(Config.BATCH_SIZE)])
    train_args.extend(["--num_clients", str(Config.NUM_CLIENTS)])
    train_args = shlex.join(train_args)  # 安全转字符串

    num_classes = read_num_classes(Path(Config.RUNTIME_DATA_ROOT) / Config.MANIFEST_PATH)

    recipe = FedAvgRecipe(
        name="convnext-fedavg",
        min_clients=Config.MIN_CLIENTS,
        num_rounds=Config.NUM_ROUNDS,
        model={
            "path": "flare_app.model.WaveletConvNeXtTiny",
            "args": {"num_classes": num_classes}
        },
        train_script="flare_app/client.py",
        train_args=train_args,
    )
    return recipe


def main():
    recipe = create_recipe()
    # add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(num_clients=Config.NUM_CLIENTS)
    run = recipe.execute(env)
    result = run.get_result()
    print("Job Status:", run.get_status())
    print()
    # SimEnv raises on execution failure; a returned result confirms completion.
    if result is None:
        raise RuntimeError("Simulation did not return a result.")
    print("Simulation completed successfully.")
    print("Result can be found in :", result)
    print()


if __name__ == "__main__":
    main()
