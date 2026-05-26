import argparse
import time
from pathlib import Path
from urllib.parse import quote

import requests


CHUNK_SIZE = 8 * 1024 * 1024


def download_file(repo_id, filename, local_dir, endpoint, expected_size=None):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    out = local_dir / filename
    tmp = out.with_suffix(out.suffix + ".part")
    url = f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{quote(filename, safe='/')}"

    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"resuming {filename} from {existing}", flush=True)
    else:
        print(f"starting {filename}", flush=True)

    downloaded = existing
    last_report = time.time()
    with requests.get(url, stream=True, timeout=(30, 120), allow_redirects=True, headers=headers) as response:
        response.raise_for_status()
        with tmp.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 30:
                    if expected_size:
                        print(f"downloaded {downloaded}/{expected_size} ({downloaded / expected_size:.1%})", flush=True)
                    else:
                        print(f"downloaded {downloaded}", flush=True)
                    last_report = now

    size = tmp.stat().st_size
    print(f"finished part size {size}", flush=True)
    if expected_size is not None and size != expected_size:
        raise SystemExit(f"size mismatch: {size} != {expected_size}")
    tmp.replace(out)
    print(out, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Directly download a Hugging Face dataset file from a mirror.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--expected-size", type=int, default=None)
    args = parser.parse_args()

    download_file(args.repo_id, args.filename, args.local_dir, args.endpoint, args.expected_size)


if __name__ == "__main__":
    main()
