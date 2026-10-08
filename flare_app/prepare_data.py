"""Prepare data."""

import shutil

from config import Config


def main():
    shutil.copytree(src="data", dst=Config.RUNTIME_DATA_ROOT, dirs_exist_ok=True)
    print(f"Synced dataset to runtime directory")


if __name__ == "__main__":
    main()
