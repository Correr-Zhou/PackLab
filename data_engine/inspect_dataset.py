"""Inspect generated packing datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def inspect_parquet(path: Path, image_root: Path | None = None, max_images: int = 20) -> dict:
    path = Path(path)
    df = pd.read_parquet(path)
    report = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "data_source_counts": _value_counts(df, "data_source"),
        "errors": [],
    }
    if "extra_info" in df.columns:
        difficulties = {}
        buffer_sizes = {}
        for info in df["extra_info"]:
            if not isinstance(info, dict):
                continue
            difficulty = info.get("difficulty")
            if difficulty is not None:
                difficulties[str(difficulty)] = difficulties.get(str(difficulty), 0) + 1
            buffer_size = info.get("buffer_size")
            if buffer_size is None:
                kwargs = info.get("interaction_kwargs", {})
                buffer_size = kwargs.get("buffer_size") if isinstance(kwargs, dict) else None
            if buffer_size is not None:
                key = str(buffer_size)
                buffer_sizes[key] = buffer_sizes.get(key, 0) + 1
        report["difficulty_counts"] = difficulties
        report["buffer_size_counts"] = buffer_sizes
    if "image_path" in df.columns:
        checked = 0
        for image_path in df["image_path"].head(max_images):
            path_obj = Path(image_path)
            if image_root is not None and not path_obj.is_absolute():
                path_obj = image_root / path_obj
            if not path_obj.exists():
                report["errors"].append(f"missing image: {path_obj}")
                continue
            with Image.open(path_obj) as img:
                if img.size != (224, 224):
                    report["errors"].append(f"unexpected image size {img.size}: {path_obj}")
            checked += 1
        report["checked_images"] = checked
    return report


def _value_counts(df, column: str) -> dict:
    if column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].value_counts().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()
    report = inspect_parquet(
        Path(args.parquet),
        image_root=Path(args.image_root) if args.image_root else None,
        max_images=args.max_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
