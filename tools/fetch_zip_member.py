#!/usr/bin/env python3
"""Extract selected files from a remote ZIP without downloading the archive.

Each ZIP entry is independently compressed and the central directory records
where it starts, so with HTTP range requests we can fetch just the bytes of the
entries we want. This lets us inspect one RCAEval fault case (~100 MB) on a
machine that cannot hold the full 8.66 GB dataset.

Usage:
    python3 tools/fetch_zip_member.py <url> <path-prefix> <out-dir>
    python3 tools/fetch_zip_member.py <url> --list
"""
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(__file__))
from probe_remote_zip import fetch, find_central_directory  # noqa: E402


def central_entries(url):
    """Yield (name, method, compressed_size, uncompressed_size, local_offset)."""
    _, cd_size, cd_off = find_central_directory(url)
    blob = fetch(url, f"{cd_off}-{cd_off + cd_size - 1}")
    off = 0
    while off + 4 <= len(blob) and blob[off:off + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", blob[off + 10:off + 12])
        csize, usize = struct.unpack("<II", blob[off + 20:off + 28])
        nlen, elen, clen = struct.unpack("<HHH", blob[off + 28:off + 34])
        local_off, = struct.unpack("<I", blob[off + 42:off + 46])
        name = blob[off + 46:off + 46 + nlen].decode("utf-8", "replace")
        yield name, method, csize, usize, local_off
        off += 46 + nlen + elen + clen


def extract(url, name, method, csize, local_off, out_path):
    # The local header repeats the name/extra lengths, which can differ from the
    # central directory's, so read it to find where the data really starts.
    head = fetch(url, f"{local_off}-{local_off + 29}")
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = local_off + 30 + nlen + elen
    data = fetch(url, f"{start}-{start + csize - 1}")
    if method == 8:                       # deflate, raw stream (no zlib header)
        data = zlib.decompress(data, -15)
    elif method != 0:                     # 0 = stored
        raise SystemExit(f"{name}: unsupported compression method {method}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)


def main():
    if len(sys.argv) == 3 and sys.argv[2] == "--list":
        for name, _, _, usize, _ in central_entries(sys.argv[1]):
            print(f"{usize / 1e6:9.2f} MB  {name}")
        return
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    url, prefix, out_dir = sys.argv[1:]
    for name, method, csize, usize, local_off in central_entries(url):
        if name.startswith(prefix) and not name.endswith("/"):
            out = os.path.join(out_dir, name)
            print(f"{usize / 1e6:8.2f} MB  {name}")
            extract(url, name, method, csize, local_off, out)


if __name__ == "__main__":
    main()
