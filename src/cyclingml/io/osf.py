"""Download files from an Open Science Framework (OSF) project.

GoldenCheetah OpenData (Liversedge 2020, doi:10.17605/OSF.IO/6HFPZ) is published on OSF as one
zip per athlete. The S3 mirror is closed and the official Python/R tools only support S3, so
this module talks to the public OSF API v2 directly (https://developer.osf.io).

Two steps, both resumable:

    # 1. list every file (all folders and sub-projects) into a manifest CSV
    python -m cyclingml.io.osf list --node 6hfpz \
        --out $CYCLINGML_DATA/raw/goldencheetah/manifest.csv

    # 2. download what the manifest lists (skips files already complete, verifies checksums)
    python -m cyclingml.io.osf download --manifest .../manifest.csv \
        --dest $CYCLINGML_DATA/raw/goldencheetah/zips --limit 100 --workers 4

Anonymous OSF access is rate limited. If you hit HTTP 429 often, create a free OSF account,
make a personal access token (Settings > Personal access tokens, scope osf.full_read) and set
it as the environment variable OSF_TOKEN.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import requests

API = "https://api.osf.io/v2"
GOLDENCHEETAH_NODE = "6hfpz"
USER_AGENT = "cyclingml-thesis/0.1 (research; +https://github.com/Kevin-ai-jpg/BachelorThesis-MachineLearning-RoadCycling)"

log = logging.getLogger(__name__)


@dataclass
class RemoteFile:
    node: str
    path: str  # e.g. "/data/033874ce-....zip"
    name: str
    size: int
    sha256: str
    md5: str
    download_url: str
    modified: str


# --------------------------------------------------------------------------- HTTP


def make_session(token: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    token = token or os.environ.get("OSF_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


class Throttle:
    """Shared pacing for all download threads.

    OSF answers HTTP 429 when we ask too fast. Waiting per thread does not help: the other
    threads keep asking. So every request first waits here: requests are spaced at least
    `min_interval` seconds apart, and after any 429 *all* threads pause until the cooldown ends.
    """

    def __init__(self, min_interval: float = 0.5) -> None:
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._cooldown_until = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot, self._cooldown_until)
            self._next_slot = start + self.min_interval
        time.sleep(max(0.0, start - time.monotonic()))

    def cool_down(self, seconds: float) -> None:
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)


THROTTLE = Throttle()


def _get(
    session: requests.Session,
    url: str,
    *,
    stream: bool = False,
    retries: int = 5,
    rate_limit_retries: int = 15,
    throttle: Throttle | None = None,
    **kw: Any,
) -> requests.Response:
    """GET with retries.

    Rate limits (429) pause every thread through the shared throttle and are retried up to
    `rate_limit_retries` times (delays double up to 5 min, ~45 min in total), because they
    always clear eventually. Network errors and 5xx get `retries` attempts.
    """
    throttle = throttle or THROTTLE
    delay, limited, failed = 2.0, 0, 0
    while True:
        throttle.wait()
        try:
            r = session.get(url, stream=stream, timeout=60, **kw)
        except (requests.ConnectionError, requests.Timeout) as e:
            failed += 1
            if failed > retries:
                raise
            log.warning("network error %s, retrying in %.0fs", e, delay)
            time.sleep(delay)
        else:
            if r.status_code < 400:
                return r
            if r.status_code == 429:
                limited += 1
                if limited > rate_limit_retries:
                    r.raise_for_status()
                wait = r.headers.get("Retry-After")
                pause = float(wait) if wait and wait.isdigit() else delay
                log.warning("rate limited (429), all downloads pause %.0fs", pause)
                throttle.cool_down(pause)
            elif r.status_code >= 500:
                failed += 1
                if failed > retries:
                    r.raise_for_status()
                log.warning("HTTP %s on %s, retrying in %.0fs", r.status_code, url, delay)
                time.sleep(delay)
            else:
                r.raise_for_status()
        delay = min(delay * 2, 300)


def _paged(session: requests.Session, url: str) -> Iterator[dict]:
    """Yield every item of a paginated OSF listing."""
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}page[size]=100"
    while url:
        body = _get(session, url).json()
        yield from body.get("data", [])
        url = (body.get("links") or {}).get("next")


# --------------------------------------------------------------------------- listing


def _to_remote_file(node: str, item: dict) -> RemoteFile:
    a = item["attributes"]
    hashes = (a.get("extra") or {}).get("hashes") or {}
    return RemoteFile(
        node=node,
        path=a.get("materialized_path") or a.get("path") or "",
        name=a["name"],
        size=int(a.get("size") or 0),
        sha256=hashes.get("sha256") or "",
        md5=hashes.get("md5") or "",
        download_url=item["links"]["download"],
        modified=a.get("date_modified") or "",
    )


def _walk_folder(session: requests.Session, node: str, url: str) -> Iterator[RemoteFile]:
    for item in _paged(session, url):
        kind = item["attributes"]["kind"]
        if kind == "file":
            yield _to_remote_file(node, item)
        elif kind == "folder":
            child = item["relationships"]["files"]["links"]["related"]["href"]
            yield from _walk_folder(session, node, child)


def list_files(
    node: str = GOLDENCHEETAH_NODE,
    session: requests.Session | None = None,
    include_children: bool = True,
) -> Iterator[RemoteFile]:
    """Every file in the node's OSF storage, recursing into folders and sub-projects."""
    session = session or make_session()
    yield from _walk_folder(session, node, f"{API}/nodes/{node}/files/osfstorage/")
    if include_children:
        for child in _paged(session, f"{API}/nodes/{node}/children/"):
            yield from list_files(child["id"], session, include_children=True)


def write_manifest(files: Iterator[RemoteFile], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    n = 0
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(RemoteFile)])
        w.writeheader()
        for rf in files:
            w.writerow(asdict(rf))
            n += 1
            if n % 500 == 0:
                log.info("listed %d files", n)
    tmp.replace(out)
    return n


def read_manifest(path: Path) -> list[RemoteFile]:
    with open(path, newline="", encoding="utf-8") as f:
        return [RemoteFile(**{**row, "size": int(row["size"])}) for row in csv.DictReader(f)]


def summarize(files: list[RemoteFile]) -> str:
    """Human-readable overview, for sending back to Claude after the list step."""
    by_ext: dict[str, list[int]] = {}
    folders: dict[str, int] = {}
    for f in files:
        ext = Path(f.name).suffix.lower() or "(none)"
        by_ext.setdefault(ext, []).append(f.size)
        folder = f"{f.node}:{f.path.rsplit('/', 1)[0] or '/'}"
        folders[folder] = folders.get(folder, 0) + 1
    lines = [f"{len(files)} files, {sum(f.size for f in files) / 1e9:.2f} GB total"]
    for ext, sizes in sorted(by_ext.items(), key=lambda kv: -sum(kv[1])):
        lines.append(f"  {ext:8} {len(sizes):6} files  {sum(sizes) / 1e9:8.2f} GB")
    lines.append("folders (node:path -> files):")
    for folder, n in sorted(folders.items(), key=lambda kv: -kv[1])[:15]:
        lines.append(f"  {folder} -> {n}")
    non_zip = [f for f in files if not f.name.lower().endswith(".zip")][:15]
    if non_zip:
        lines.append("non-zip files:")
        lines += [f"  {f.node}:{f.path} ({f.size / 1e6:.1f} MB)" for f in non_zip]
    return "\n".join(lines)


# --------------------------------------------------------------------------- download


def _hash_ok(path: Path, rf: RemoteFile) -> bool:
    if rf.sha256:
        h, want = hashlib.sha256(), rf.sha256
    elif rf.md5:
        h, want = hashlib.md5(), rf.md5  # noqa: S324 - checksum, not security
    else:
        return True
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == want


def is_complete(path: Path, rf: RemoteFile) -> bool:
    return path.exists() and (rf.size == 0 or path.stat().st_size == rf.size)


def download_one(session: requests.Session, rf: RemoteFile, dest: Path, verify: bool = True) -> str:
    """Download one file to dest/<name>. Returns 'skipped', 'downloaded' or raises."""
    target = dest / rf.name
    if is_complete(target, rf):
        return "skipped"
    part = target.with_name(target.name + ".part")
    with _get(session, rf.download_url, stream=True) as r, open(part, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    if rf.size and part.stat().st_size != rf.size:
        part.unlink(missing_ok=True)
        raise OSError(f"{rf.name}: size mismatch")
    if verify and not _hash_ok(part, rf):
        part.unlink(missing_ok=True)
        raise OSError(f"{rf.name}: checksum mismatch")
    part.replace(target)
    return "downloaded"


def download(
    files: list[RemoteFile],
    dest: Path,
    workers: int = 4,
    limit: int | None = None,
    suffix: str = ".zip",
    session: requests.Session | None = None,
) -> dict[str, int]:
    """Download (up to `limit`) files ending in `suffix`, in manifest order, in parallel."""
    dest.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if f.name.lower().endswith(suffix)] if suffix else list(files)
    if limit is not None:
        todo = todo[:limit]
    session = session or make_session()
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failed_log = dest / "failed.txt"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, session, rf, dest): rf for rf in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rf = futures[fut]
            try:
                counts[fut.result()] += 1
            except Exception as e:  # keep going; failures are listed for a rerun
                counts["failed"] += 1
                log.error("failed %s: %s", rf.name, e)
                with open(failed_log, "a", encoding="utf-8") as fl:
                    fl.write(f"{rf.name}\t{e}\n")
            if i % 50 == 0 or i == len(todo):
                log.info("%d/%d  %s", i, len(todo), counts)
    return counts


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="write a manifest of every file in an OSF project")
    pl.add_argument("--node", default=GOLDENCHEETAH_NODE)
    pl.add_argument("--out", type=Path, required=True)
    pd = sub.add_parser("download", help="download the files listed in a manifest")
    pd.add_argument("--manifest", type=Path, required=True)
    pd.add_argument("--dest", type=Path, required=True)
    pd.add_argument("--limit", type=int, default=None, help="only the first N files")
    pd.add_argument("--workers", type=int, default=3)
    pd.add_argument(
        "--min-interval", type=float, default=0.5, help="seconds between requests (all threads)"
    )
    pd.add_argument("--suffix", default=".zip", help="file suffix to download ('' = all)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "list":
        n = write_manifest(list_files(args.node), args.out)
        print(f"Wrote {n} entries to {args.out}\n")
        print(summarize(read_manifest(args.out)))
    else:
        THROTTLE.min_interval = args.min_interval
        counts = download(
            read_manifest(args.manifest), args.dest, args.workers, args.limit, args.suffix
        )
        print(counts)


if __name__ == "__main__":
    main()
