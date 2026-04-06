"""
Generate placeholder PNG icons for the Chrome extension.
Run: python extension/generate_icons.py
"""

import struct
import zlib
from pathlib import Path


def create_png(width, height, r, g, b):
    """Create a simple solid-color PNG file as bytes."""
    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    # IDAT — raw pixel data
    raw = b""
    for y in range(height):
        raw += b"\x00"  # filter byte
        for x in range(width):
            # Simple "G" letter shape in the center
            cx, cy = width // 2, height // 2
            radius = width // 3
            dx, dy = x - cx, y - cy
            dist = (dx * dx + dy * dy) ** 0.5

            # Ring shape
            in_ring = radius * 0.6 < dist < radius
            # Cut out right side for G opening
            in_opening = dx > 0 and -radius * 0.2 < dy < radius * 0.2
            # Horizontal bar of G
            in_bar = dx > 0 and abs(dy) < radius * 0.15 and dist < radius

            if (in_ring and not in_opening) or (in_bar and dx < radius * 0.5):
                raw += bytes([255, 255, 255])  # white G
            else:
                raw += bytes([r, g, b])  # background

    compressed = zlib.compress(raw)
    idat = make_chunk(b"IDAT", compressed)

    # IEND
    iend = make_chunk(b"IEND", b"")

    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def main():
    icons_dir = Path(__file__).parent / "icons"
    icons_dir.mkdir(exist_ok=True)

    # Green-teal color (matches the Genzon brand)
    r, g, b = 15, 110, 86  # #0F6E56

    for size in [16, 48, 128]:
        data = create_png(size, size, r, g, b)
        path = icons_dir / f"icon{size}.png"
        path.write_bytes(data)
        print(f"  ✓ Created {path} ({size}x{size})")

    print("\n  Done! Icons ready.")


if __name__ == "__main__":
    main()