import argparse
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import requests


REPO_ID = "nasa-impact/WxC-Bench"
REVISION = "main"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_LOCAL_DIR = "/home/daxiniu12/Synoptic-Bench/data/wx"
DEFAULT_PATH = "long_term_precipitation_forecast"
CHUNK_SIZE = 1024 * 1024


def parse_next_cursor(link_header):
    if not link_header:
        return None

    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        start = section.find("<")
        end = section.find(">")
        if start == -1 or end == -1:
            continue
        query = parse_qs(urlparse(section[start + 1 : end]).query)
        cursors = query.get("cursor")
        if cursors:
            return cursors[0]
    return None


def tree_url(endpoint, path_in_repo, cursor=None, limit=1000):
    encoded_path = quote(path_in_repo.strip("/"), safe="/")
    url = (
        f"{endpoint.rstrip('/')}/api/datasets/{REPO_ID}/tree/{REVISION}/"
        f"{encoded_path}?recursive=true&expand=false&limit={limit}"
    )
    if cursor:
        url = f"{url}&cursor={quote(cursor, safe='')}"
    return url


def iter_repo_files(endpoint, path_in_repo):
    session = requests.Session()
    cursor = None
    while True:
        response = session.get(tree_url(endpoint, path_in_repo, cursor=cursor), timeout=60)
        response.raise_for_status()
        for item in response.json():
            if item.get("type") == "file":
                yield item["path"], item.get("size")

        cursor = parse_next_cursor(response.headers.get("link"))
        if not cursor:
            break


def local_path_for(local_dir, repo_path):
    return Path(local_dir) / repo_path


def file_is_present(local_dir, repo_path, expected_size=None):
    path = local_path_for(local_dir, repo_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return False
    if expected_size is None:
        return True
    return path.stat().st_size == expected_size


def download_file(endpoint, local_dir, repo_path, expected_size=None, retries=8):
    output_path = local_path_for(local_dir, repo_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    url = f"{endpoint.rstrip('/')}/datasets/{REPO_ID}/resolve/{REVISION}/{quote(repo_path, safe='/')}"

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(15, 20), allow_redirects=True) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)

            if expected_size is not None and tmp_path.stat().st_size != expected_size:
                raise IOError(f"size mismatch: got {tmp_path.stat().st_size}, expected {expected_size}")

            tmp_path.replace(output_path)
            return
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def main():
    parser = argparse.ArgumentParser(
        description="Download WxC-Bench long_term_precipitation_forecast files via hf-mirror, one file at a time."
    )
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--path", default=DEFAULT_PATH, help="Repo subdirectory to download.")
    parser.add_argument("--limit", type=int, default=None, help="Download at most N missing files.")
    parser.add_argument("--force", action="store_true", help="Redownload files even when present locally.")
    args = parser.parse_args()

    files = sorted(iter_repo_files(args.endpoint, args.path))
    print(f"Found {len(files)} files under {args.path}", flush=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for repo_path, expected_size in files:
        if not args.force and file_is_present(args.local_dir, repo_path, expected_size):
            skipped += 1
            continue

        if args.limit is not None and downloaded >= args.limit:
            break

        try:
            download_file(args.endpoint, args.local_dir, repo_path, expected_size)
            downloaded += 1
            if downloaded % 10 == 0 or downloaded == 1:
                print(f"Downloaded {downloaded} files; skipped {skipped}; latest: {repo_path}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAILED {repo_path}: {exc}", flush=True)

    print(f"Done. downloaded={downloaded}, skipped={skipped}, failed={failed}, path={args.path}", flush=True)


if __name__ == "__main__":
    main()
