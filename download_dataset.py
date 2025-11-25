import os
import tarfile
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

REPO_ID = "ChiWei085/ML2025_HW_Cardiac_Muscle"
SPLITS: tuple[str, ...] = ("train", "val", "test")
DATA_ROOT = Path("datasets")


def get_hf_token() -> str | None:
    load_dotenv(override=False)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        print("Using Hugging Face token from environment.")
    else:
        print("No Hugging Face token found; downloading with public access.")
    return token


def download_and_extract(
    repo_id: str, split_names: Iterable[str], root: Path, token: str | None = None
) -> None:
    root.mkdir(exist_ok=True)
    for split in split_names:
        fname = f"{split}.tar.gz"
        print(f"Downloading {fname} from {repo_id} ...")

        dataset = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            token=token,
        )

        print(f"Extracting {fname} -> {root} ...")
        with tarfile.open(dataset, "r:gz") as tar:
            tar.extractall(path=root)


def main() -> None:
    token = get_hf_token()
    download_and_extract(REPO_ID, SPLITS, DATA_ROOT, token=token)


if __name__ == "__main__":
    main()
