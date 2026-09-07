#!/usr/bin/env python3
"""Measure a remote ZIP's true uncompressed size without downloading it.

A ZIP stores a "central directory" at the end of the file listing every entry
with its compressed AND uncompressed size. If the server supports HTTP range
requests we can fetch just that trailer (a few hundred KB) and read the exact
extracted footprint of a multi-GB archive.

Used to size the RCAEval RE2-OB dataset (1.19 GB download -> 8.66 GB on disk)
against available storage, without spending the bandwidth or the disk.

Usage:  python3 tools/probe_remote_zip.py <url>
"""
import collections
import struct
import subprocess
import sys


def fetch(url: str, byte_range: str) -> bytes:
    """Fetch a byte range via curl. `byte_range` uses curl -r syntax."""
    out = subprocess.run(
        ["curl", "-sfL", "-r", byte_range, url],
        capture_output=True, check=True,
    )
    return out.stdout


def find_central_directory(url: str):
    """Locate the central directory by parsing the End Of Central Directory record."""
    tail = fetch(url, "-65536")
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise SystemExit("no EOCD record found - not a zip, or trailer too large")
    n_entries, cd_size, cd_off = struct.unpack("<HII", tail[i + 10 : i + 20])
    # A zip64 archive signals overflow with sentinel 0xFFFFFFFF values.
    if 0xFFFFFFFF in (cd_size, cd_off):
        j = tail.rfind(b"PK\x06\x06")
        if j < 0:
            raise SystemExit("zip64 archive but no zip64 EOCD found")
        n_entries, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32 : j + 56])
    return n_entries, cd_size, cd_off


def parse_entries(blob: bytes):
    """Walk central-directory headers, yielding (name, compressed, uncompressed)."""
    off = 0
    while off + 4 <= len(blob) and blob[off : off + 4] == b"PK\x01\x02":
        csize, usize = struct.unpack("<II", blob[off + 20 : off + 28])
        nlen, elen, clen = struct.unpack("<HHH", blob[off + 28 : off + 34])
        name = blob[off + 46 : off + 46 + nlen].decode("utf-8", "replace")
        yield name, csize, usize
        off += 46 + nlen + elen + clen


def main(url: str) -> None:
    n_entries, cd_size, cd_off = find_central_directory(url)
    entries = list(parse_entries(fetch(url, f"{cd_off}-{cd_off + cd_size - 1}")))
    print(f"entries: {len(entries)} (header claimed {n_entries})")

    comp = sum(e[1] for e in entries)
    uncomp = sum(e[2] for e in entries)
    print(f"compressed  : {comp / 1e9:6.2f} GB")
    print(f"uncompressed: {uncomp / 1e9:6.2f} GB")
    print(f"expansion   : {uncomp / max(comp, 1):6.1f}x")

    # Group by basename: in RCAEval every fault case holds the same filenames,
    # so this shows which modality dominates the footprint.
    count, size = collections.Counter(), collections.Counter()
    for name, _, usize in entries:
        if name.endswith("/"):
            continue
        base = name.split("/")[-1]
        count[base] += 1
        size[base] += usize

    print("\nby filename:")
    for base, n in count.most_common(15):
        print(f"  {base:<28} n={n:<5} {size[base] / 1e9:7.2f} GB"
              f"  avg={size[base] / n / 1e6:7.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
