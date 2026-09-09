#!/usr/bin/env python3
"""Upload the baked checkpoint to the Hub.

The token is read by huggingface_hub from its own on-disk store (~/.cache/huggingface/token,
written by `huggingface-cli login`). It is never passed as an argument and never printed.
"""
import argparse, os, sys
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

api = HfApi()                      # token resolved from the hub's own store, not from us
files = sorted(os.listdir(a.dir))
size = sum(os.path.getsize(os.path.join(a.dir, f)) for f in files)
print(f"  {len(files)} files, {size/1e9:.2f} GB -> {a.repo}")
if a.dry_run:
    print("  dry run, not uploading"); sys.exit(0)
api.create_repo(a.repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=a.dir, repo_id=a.repo, repo_type="model",
                  commit_message="Bake lm_head + MTP draft to GPTQ INT4 on disk")
print("  uploaded")
