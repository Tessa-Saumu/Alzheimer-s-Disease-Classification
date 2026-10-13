"""Audit a permitted image split manifest; hash equality is not patient identity."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def audit(manifest: Path) -> dict:
    """Read CSV path,split rows (paths relative to manifest); return collision evidence.

    Splits must be train/validation/test. Exact SHA256 hashes concern file bytes;
    8x8 grayscale average hashes only flag visually similar candidates. No
    patient-level or original experimental leakage percentage is inferred.
    """
    raw = manifest.read_bytes()
    groups = {"sha256": defaultdict(list), "average_hash_8x8": defaultdict(list)}
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if not rows:
        raise ValueError("Manifest contains no images")
    seen = set()
    census = []
    for row in rows:
        relative, split = row["path"], row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        path = (manifest.parent / relative).resolve()
        if path in seen:
            raise ValueError(f"Repeated image path: {relative}")
        seen.add(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((8, 8), Image.Resampling.LANCZOS).get_flattened_data())
        mean = sum(pixels) / len(pixels)
        perceptual = f"{sum((pixel > mean) << i for i, pixel in enumerate(pixels)):016x}"
        item = {"path": relative, "split": split, "sha256": digest}
        census.append(item)
        groups["sha256"][digest].append(item)
        groups["average_hash_8x8"][perceptual].append(item)
    collisions = {method: [{"hash": key, "images": values}
                           for key, values in sorted(buckets.items())
                           if len({item["split"] for item in values}) > 1]
                  for method, buckets in groups.items()}
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "images": census,
            "cross_split_groups": collisions,
            "interpretation": "Candidate duplication evidence; not patient identity or clinical performance."}


def main() -> None:
    """Parse an explicit manifest and save a JSON evidence record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("audit_result.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.manifest), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
