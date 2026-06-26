from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload prepared wheels to a Hugging Face dataset repo.")
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Update qwentts-cpp-python wheels")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        folder_path=str(args.folder),
        path_in_repo=".",
        commit_message=args.commit_message,
    )
    print(f"Uploaded {args.folder} to dataset {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
