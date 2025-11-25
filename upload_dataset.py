import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ID = "ChiWei085/ML2025_HW_Cardiac_Muscle"
DATA_ROOT = Path("datasets")


def main() -> None:
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("[ENV] HF_TOKEN is needed.")

    api = HfApi(token=token)

    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
    )

    if not DATA_ROOT.is_dir():
        raise RuntimeError(f"[ERROR] {DATA_ROOT} not found.")

    for fname in ("train.tar.gz", "val.tar.gz", "test.tar.gz"):
        compress_dataset = DATA_ROOT / fname
        if not compress_dataset.is_file():
            print(f"[WARN] {compress_dataset} not found, skip")
            continue

        print(f"[INFO] upload {compress_dataset} -> {REPO_ID}/{fname}")
        api.upload_file(
            path_or_fileobj=str(compress_dataset),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    print(f"[DONE] Uploaded to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
