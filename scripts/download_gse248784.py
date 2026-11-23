
#!/usr/bin/env python3
"""Resumable parallel Range downloader for GSE248784_RAW.tar.

NCBI throttles a single connection to ~12 KB/s in this sandbox and direct
egress times out; the local HTTP proxy is required. Parallel Range requests
scale to ~0.7-0.8 MB/s aggregate. Chunks are small (default 1 MB) so each
request finishes well inside the proxy idle timeout.

Usage:
  python download_gse248784.py [--workers 64] [--chunk-mb 1]
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE248nnn/"
       "GSE248784/suppl/GSE248784_RAW.tar")
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "http://127.0.0.1:18080"
BASE = os.path.join(_ROOT, "data", "external", "gse248784")
PARTS = os.path.join(BASE, "parts")
OUT = os.path.join(BASE, "GSE248784_RAW.tar")
EXPECTED = 1977128960

os.makedirs(PARTS, exist_ok=True)
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def head_size():
    try:
        r = requests.head(URL, proxies={"http": PROXY, "https": PROXY},
                          timeout=(15, 60), allow_redirects=True)
        if r.status_code == 200 and r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    except Exception as e:
        log(f"HEAD failed: {e}")
    return EXPECTED


def chunk_path(i):
    return os.path.join(PARTS, f"chunk_{i:05d}")


def fetch(i, start, end, size, session):
    path = chunk_path(i)
    expected = end - start + 1
    if os.path.exists(path) and os.path.getsize(path) == expected:
        return "skip", expected
    tmp = path + ".tmp"
    last = None
    for attempt in range(12):
        try:
            headers = {"Range": f"bytes={start}-{end}",
                       "Accept-Encoding": "identity",
                       "User-Agent": "curl/8.5.0"}
            with session.get(URL, headers=headers,
                             proxies={"http": PROXY, "https": PROXY},
                             stream=True, timeout=(15, 180)) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"HTTP {r.status_code} for {start}-{end}")
                cr = r.headers.get("Content-Range", "")
                if not cr.startswith(f"bytes {start}-"):
                    raise RuntimeError(f"bad Content-Range: {cr}")
                got = 0
                with open(tmp, "wb") as fh:
                    for block in r.iter_content(65536):
                        if block:
                            fh.write(block)
                            got += len(block)
                if got != expected:
                    raise RuntimeError(f"short read {got}/{expected}")
            os.replace(tmp, path)
            return "ok", expected
        except Exception as e:
            last = e
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            time.sleep(min(30, 2 ** attempt))
    return "fail", last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk-mb", type=int, default=1)
    args = ap.parse_args()

    size = head_size()
    chunk = args.chunk_mb * 1024 * 1024
    n = (size + chunk - 1) // chunk
    log(f"size={size} chunks={n} chunk={chunk} workers={args.workers} proxy={PROXY}")

    tasks = []
    for i in range(n):
        s = i * chunk
        e = min(size - 1, s + chunk - 1)
        tasks.append((i, s, e, size))

    done_bytes = 0
    for i, s, e, _ in tasks:
        p = chunk_path(i)
        if os.path.exists(p) and os.path.getsize(p) == e - s + 1:
            done_bytes += e - s + 1
    log(f"already complete: {done_bytes}/{size} ({done_bytes/size*100:.1f}%)")

    pending = [t for t in tasks
               if not (os.path.exists(chunk_path(t[0])) and
                       os.path.getsize(chunk_path(t[0])) == t[2] - t[1] + 1)]

    t0 = time.time()
    base_done = done_bytes
    failures = []
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers,
                                            pool_maxsize=args.workers,
                                            max_retries=0)
    session.mount("https://", adapter)

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch, i, s, e, size, session): (i, s, e)
                    for i, s, e, _ in pending}
            last_log = time.time()
            for k, fut in enumerate(as_completed(futs), 1):
                status, info = fut.result()
                i, s, e = futs[fut]
                if status == "ok":
                    done_bytes += e - s + 1
                elif status == "fail":
                    failures.append((i, s, e, str(info)))
                now = time.time()
                if now - last_log >= 20 or k == len(futs):
                    elapsed = max(now - t0, 1e-9)
                    rate = (done_bytes - base_done) / elapsed
                    remain = size - done_bytes
                    eta = remain / rate if rate > 0 else 0
                    log(f"{k}/{len(futs)} chunks | {done_bytes/size*100:5.1f}% | "
                        f"{rate/1024:.0f} KB/s | ETA {eta/60:.1f} min | fail {len(failures)}")
                    last_log = now

    if failures:
        log(f"FAILED chunks: {len(failures)}; first: {failures[:3]}")
        with open(os.path.join(BASE, "download_failures.json"), "w") as fh:
            json.dump(failures, fh)
        return 1

    log("all chunks complete; concatenating")
    tmp_out = OUT + ".tmp"
    with open(tmp_out, "wb") as out:
        for i in range(n):
            with open(chunk_path(i), "rb") as fh:
                shutil.copyfileobj(fh, out, length=8 * 1024 * 1024)
    os.replace(tmp_out, OUT)
    got = os.path.getsize(OUT)
    log(f"assembled {OUT}: {got} bytes (expected {size})")
    with open(os.path.join(BASE, "download_complete.json"), "w") as fh:
        json.dump({"url": URL, "size": got, "expected": size,
                   "workers": args.workers, "chunk_mb": args.chunk_mb,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)
    return 0 if got == size else 1


if __name__ == "__main__":
    sys.exit(main())
