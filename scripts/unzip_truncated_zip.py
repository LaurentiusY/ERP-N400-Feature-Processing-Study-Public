"""Extract a >4 GB zip whose central-directory offsets are truncated to 32 bits.

macOS archives larger than 4 GB are often written without zip64 records, so the stored
local-header offsets wrap at 4 GiB and standard tools (Python zipfile, Windows tar) fail
("Bad magic number", "Damaged Zip archive"). Used for the Draschkow et al. (2018) dataset.

For each central-directory entry, the real offset is the stored 32-bit value plus the smallest
multiple of 2**32 that keeps the offsets increasing; the local header signature and file name
are checked there, and every file's CRC-32 and size are checked after decompression. macOS
metadata (__MACOSX, .DS_Store) is skipped.

    python scripts/unzip_truncated_zip.py ZIP OUTDIR
"""

import struct
import sys
import zipfile
import zlib
from pathlib import Path

GIB4 = 1 << 32


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    z = zipfile.ZipFile(src)
    infos = z.infolist()
    concat = infos[0].header_offset  # first entry is stored at offset 0; zipfile added this shift
    prev_end, ok, skipped = 0, 0, 0
    with open(src, "rb") as f:
        for info in infos:
            stored = (info.header_offset - concat) % GIB4
            real = stored + (prev_end // GIB4) * GIB4
            if real < prev_end:
                real += GIB4
            f.seek(real)
            head = f.read(30)
            sig, *_ = struct.unpack("<I", head[:4])
            if sig != 0x04034B50:
                raise SystemExit(f"no local header for {info.filename} at {real}")
            n_name, n_extra = struct.unpack("<HH", head[26:30])
            name = f.read(n_name).decode("utf-8", "replace")
            if name != info.filename:
                raise SystemExit(f"name mismatch at {real}: {name!r} != {info.filename!r}")
            f.seek(n_extra, 1)
            prev_end = f.tell() + info.compress_size
            if (
                info.filename.startswith("__MACOSX")
                or info.filename.endswith(".DS_Store")
                or info.is_dir()
            ):
                skipped += 1
                continue
            target = out / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            dec = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
            crc, left = 0, info.compress_size
            with open(target, "wb") as g:
                while left:
                    block = f.read(min(left, 1 << 24))
                    left -= len(block)
                    data = dec.decompress(block) if dec else block
                    crc = zlib.crc32(data, crc)
                    g.write(data)
                if dec:
                    tail = dec.flush()
                    crc = zlib.crc32(tail, crc)
                    g.write(tail)
            if crc != info.CRC or target.stat().st_size != info.file_size:
                raise SystemExit(f"CRC/size mismatch for {info.filename}")
            ok += 1
    print(f"extracted {ok} files (CRC-32 and size checked), skipped {skipped} entries")


if __name__ == "__main__":
    main()
