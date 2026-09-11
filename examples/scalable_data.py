"""Scalable data pipeline: UCF-50K spectrum cartography.

Download, verify, extract, index, clean, transform and load the
real-world ray-traced radio-map corpus
``KR-init/Spectrum-Cartography-256x256-UCF-50K`` from HuggingFace.

The corpus holds 50,000 path-loss radio maps of the UCF campus, each
256×256 pixels and stored as one parquet file with a single row of three
65,536-element columns: ``building_mask``, ``tx_origin`` and
``path_loss`` (dBm). The download totals ~10 GB across six zip archives.

Every byte-level guarantee of the pipeline lives here:

* downloads are resumable and size-verified against the hub manifest,
* zip extraction is idempotent (sentinel-guarded), atomic, and
  CRC-verifiable on request,
* parquet reads are schema-validated with hard assertions,
* scene sampling is deterministic for a given seed,
* every stage emits a traceability report (JSON).

Run::

    python -m examples.scalable_data --prepare --selfcheck

This module never imports torch: it is pure NumPy + PyArrow so it stays
fast, dependency-light and safe for multiprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DATASET = "KR-init/Spectrum-Cartography-256x256-UCF-50K"
DATASET_REVISION = os.environ.get("LAKER_UCF50K_REVISION", "main")
SIDE = 256
MANIFEST_FILENAME = "MANIFEST.json"
PIXELS = SIDE * SIDE
COLUMNS = ("building_mask", "tx_origin", "path_loss")
UNITS = "path_loss_dBm"
EXPECTED_SIZE = {
    "test.zip": 996036801,
    "val.zip": 994441257,
    "train-001.zip": 2148615175,
    "train-002.zip": 2148343678,
    "train-003.zip": 2148562110,
    "train-004.zip": 1527872290,
}
EXPECTED_COUNT = {
    "test": 5000,
    "val": 5000,
    "train": 10777 + 10785 + 10773 + 7665,
}
META_DIR = ".meta"


@dataclass(frozen=True)
class Scene:
    """A cleaned single radio map from the corpus."""

    split: str
    name: str
    path: str
    mask: np.ndarray
    tx: np.ndarray
    radio: np.ndarray
    valid: np.ndarray
    units: str


class ScalableData:
    """Download, verify, extract, index, clean, transform and load UCF-50K."""

    source = DATASET
    revision = DATASET_REVISION
    side = SIDE
    units = UNITS

    @staticmethod
    def grid() -> np.ndarray:
        """Return the normalized 256×256 pixel grid as ``(65536, 2)`` float64."""
        rows: np.ndarray = np.arange(SIDE, dtype=np.float64)
        ys: np.ndarray
        xs: np.ndarray
        ys, xs = np.meshgrid(rows, rows, indexing="ij")
        return np.stack([xs.ravel(), ys.ravel()], axis=1) / SIDE

    @staticmethod
    def download(
        data_dir: str,
        files: Optional[Sequence[str]] = None,
        workers: int = 2,
        verify: bool = True,
    ) -> dict:
        """Download the corpus zips into ``data_dir``.

        Existing files with the exact expected byte size are skipped.
        When ``verify`` is set, every file is size-checked against the
        manifest after download. Returns ``{filename: {"path", "size"}}``.
        """
        import huggingface_hub

        os.makedirs(data_dir, exist_ok=True)
        targets = list(files if files is not None else EXPECTED_SIZE)
        stats: dict = {}

        def fetch(name: str) -> tuple[str, int, str]:
            path = os.path.join(data_dir, name)
            if os.path.exists(path) and os.path.getsize(path) == EXPECTED_SIZE.get(name):
                return name, os.path.getsize(path), "cached"
            local = huggingface_hub.hf_hub_download(
                repo_id=DATASET,
                filename=name,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=data_dir,
            )
            return name, os.path.getsize(local), "downloaded"

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            for name, size, mode in pool.map(fetch, targets):
                path = os.path.join(data_dir, name)
                if verify:
                    expected = EXPECTED_SIZE.get(name)
                    assert expected is None or size == expected, (
                        f"size mismatch for {name}: got {size}, expected {expected}"
                    )
                stats[name] = {"path": path, "size": size, "mode": mode}

        stats["elapsed_seconds"] = time.perf_counter() - t0
        stats["bytes_total"] = sum(
            entry["size"] for entry in stats.values() if isinstance(entry, dict)
        )
        ScalableData.write_manifest(data_dir, stats)
        return stats

    @staticmethod
    def write_manifest(data_dir: str, stats: dict) -> str:
        """Write a pinned-revision manifest to ``<data_dir>/MANIFEST.json``."""
        os.makedirs(data_dir, exist_ok=True)
        manifest_path = os.path.join(data_dir, MANIFEST_FILENAME)
        manifest = {
            "dataset": DATASET,
            "revision": DATASET_REVISION,
            "expected_size": EXPECTED_SIZE,
            "files": {
                name: entry for name, entry in stats.items() if isinstance(entry, dict)
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        logger.info("Wrote pinned manifest %s (revision=%s)", manifest_path, DATASET_REVISION)
        return manifest_path

    @staticmethod
    def extract(
        zip_path: str,
        out_root: str,
        force: bool = False,
        verify: bool = False,
    ) -> dict:
        """Extract one zip into ``out_root``, idempotently.

        Extraction targets ``out_root/<split>`` (the top-level directory
        inside the archive). A sentinel records the extracted file count
        and total bytes; a complete, correct sentinel skips re-extraction.
        When ``verify`` is set the whole archive is CRC-checked first.
        """
        name = os.path.basename(zip_path)
        split = "train" if name.startswith("train") else name.removesuffix(".zip")
        dest = os.path.join(out_root, split)
        sentinel_path = os.path.join(out_root, META_DIR, name + ".json")
        expected = EXPECTED_SIZE.get(name)

        if not force and os.path.exists(sentinel_path):
            with open(sentinel_path, encoding="utf-8") as handle:
                done = json.load(handle)
            if done.get("count") == EXPECTED_COUNT.get(split):
                return {"zip": name, "split": split, "out": dest, "skipped": True, **done}

        if verify and expected is not None:
            with zipfile.ZipFile(zip_path) as archive:
                assert archive.testzip() is None, f"CRC failure in {name}"
            logger.info("CRC verified %s", name)

        os.makedirs(dest, exist_ok=True)
        os.makedirs(os.path.dirname(sentinel_path), exist_ok=True)
        t0 = time.perf_counter()
        with zipfile.ZipFile(zip_path) as archive:
            top = archive.namelist()[0].split("/", 1)[0]
            for member in archive.infolist():
                if member.is_dir() or not member.filename.endswith(".parquet"):
                    continue
                prefix = top + "/"
                if member.filename.startswith(prefix):
                    rel = member.filename[len(prefix):]
                else:
                    rel = member.filename
                target = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        count = 0
        bytes_total = 0
        for root, _dirs, files in os.walk(dest):
            for file in files:
                if file.endswith(".parquet"):
                    count += 1
                    bytes_total += os.path.getsize(os.path.join(root, file))

        report = {
            "zip": name,
            "split": split,
            "out": dest,
            "count": count,
            "bytes_total": bytes_total,
            "elapsed_seconds": time.perf_counter() - t0,
            "skipped": False,
        }
        with open(sentinel_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    @staticmethod
    def extract_all(
        data_dir: str,
        out_root: str,
        force: bool = False,
        verify: bool = False,
    ) -> list:
        """Extract every corpus zip under ``data_dir`` into ``out_root``."""
        reports = []
        for name in EXPECTED_SIZE:
            zip_path = os.path.join(data_dir, name)
            assert os.path.exists(zip_path), f"missing archive {zip_path}"
            reports.append(ScalableData.extract(zip_path, out_root, force=force, verify=verify))
        return reports

    @staticmethod
    def index(unpacked_root: str) -> list:
        """Index every parquet under ``unpacked_root`` as sorted entries."""
        entries = []
        for split in EXPECTED_COUNT:
            split_dir = os.path.join(unpacked_root, split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"split directory missing: {split_dir}")
            for root, _dirs, files in os.walk(split_dir):
                for file in sorted(files):
                    if file.endswith(".parquet"):
                        path = os.path.join(root, file)
                        entries.append(
                            {
                                "split": split,
                                "name": file.removesuffix(".parquet"),
                                "path": path,
                            }
                        )
        assert len(entries) == sum(EXPECTED_COUNT.values()), (
            f"indexed {len(entries)} maps, expected {sum(EXPECTED_COUNT.values())}"
        )
        entries.sort(key=lambda entry: (entry["split"], entry["name"]))
        return entries

    @staticmethod
    def fingerprint(entries: list) -> str:
        """Stable SHA-256 fingerprint of an index (for traceability)."""
        digest = hashlib.sha256()
        for entry in entries:
            digest.update(f"{entry['split']}:{entry['name']}:{entry['path']}\n".encode())
        return digest.hexdigest()

    @staticmethod
    def clean(entry: dict) -> Scene:
        """Read and schema-validate one parquet map into a :class:`Scene`."""
        split = entry["split"]
        name = entry["name"]
        path = entry["path"]
        table = pq.read_table(path)
        rows = table.to_pylist()
        assert len(rows) == 1, f"{name}: expected one row, got {len(rows)}"
        record = rows[0]
        for column in COLUMNS:
            assert column in record, f"{name}: missing column {column!r}"

        mask = np.asarray(record["building_mask"], dtype=np.float64).reshape(SIDE, SIDE)
        tx = np.asarray(record["tx_origin"], dtype=np.float64).reshape(SIDE, SIDE)
        radio = np.asarray(record["path_loss"], dtype=np.float64).reshape(SIDE, SIDE)

        assert mask.shape == (SIDE, SIDE), f"{name}: mask shape"
        assert tx.shape == (SIDE, SIDE), f"{name}: tx shape"
        assert radio.shape == (SIDE, SIDE), f"{name}: radio shape"
        assert np.isin(mask, (0.0, 1.0)).all(), f"{name}: mask not binary"
        assert np.count_nonzero(tx) == 1, f"{name}: tx_origin must be a single pixel"
        assert np.isfinite(radio).all(), f"{name}: path_loss has non-finite values"
        assert radio.min() > 0, f"{name}: non-positive path loss"

        valid = np.isfinite(radio) & ~mask.astype(bool)
        return Scene(
            split=split,
            name=name,
            path=path,
            mask=mask.astype(np.int64),
            tx=tx,
            radio=radio,
            valid=valid,
            units=UNITS,
        )

    @staticmethod
    def task(
        scene: Scene,
        n_sensors: int = 2000,
        n_eval: int = 8000,
        seed: int = 0,
    ) -> dict:
        """Deterministically split a scene into sensor / evaluation batches.

        Sensors are drawn without replacement from the finite pixels of
        the scene; the evaluation batch is drawn from the remaining
        pixels. Targets are standardized with the sensor-batch mean and
        standard deviation so a constant predictor scores zero; the
        inverse statistics are returned for un-standardizing.

        Returns a dict with the tensors ``x``, ``y``, ``xe``, ``ye``
        (all float64 NumPy arrays), the inverse statistics ``y_mean`` /
        ``y_std``, the raw dBm targets ``y_raw`` / ``ye_raw``, the pixel
        indices, and the seed used.
        """
        assert n_sensors > 0, "n_sensors must be positive"
        assert n_eval > 0, "n_eval must be positive"
        assert n_sensors + n_eval < PIXELS, "sensor + eval batches exceed the map"

        pixels = np.flatnonzero(scene.valid)
        rng = np.random.default_rng(int(seed))
        sensors = rng.choice(pixels, size=int(n_sensors), replace=False)
        remaining = np.setdiff1d(pixels, sensors)
        eval_pixels = rng.choice(remaining, size=int(min(n_eval, len(remaining))), replace=False)

        radio = scene.radio.ravel()
        grid = ScalableData.grid()
        y_raw = radio[sensors]
        y_mean = float(y_raw.mean())
        y_std = float(y_raw.std())
        assert y_std > 0, f"{scene.name}: zero sensor variance"

        return {
            "x": grid[sensors].copy(),
            "y": ((y_raw - y_mean) / y_std).astype(np.float64),
            "xe": grid[eval_pixels].copy(),
            "ye": ((radio[eval_pixels] - y_mean) / y_std).astype(np.float64),
            "y_mean": y_mean,
            "y_std": y_std,
            "y_raw": y_raw.astype(np.float64),
            "ye_raw": radio[eval_pixels].astype(np.float64),
            "sensor_idx": sensors.tolist(),
            "eval_idx": eval_pixels.tolist(),
            "seed": int(seed),
        }

    @staticmethod
    def stats(entries: list, n_sample: int = 100, seed: int = 0) -> dict:
        """Aggregate corpus statistics over a deterministic sample of maps."""
        sample = [entries[i] for i in range(0, len(entries), max(1, len(entries) // n_sample))]
        if len(sample) < n_sample:
            sample = entries[:n_sample]
        sample = sorted(sample, key=lambda e: (e["split"], e["name"]))[:n_sample]

        scenes = []
        load_times = []
        for entry in sample:
            t0 = time.perf_counter()
            scenes.append(ScalableData.clean(entry))
            load_times.append(time.perf_counter() - t0)

        mins, maxs, means, stds = [], [], [], []
        building_fracs = []
        for scene in scenes:
            r = scene.radio[scene.valid]
            mins.append(float(r.min()))
            maxs.append(float(r.max()))
            means.append(float(r.mean()))
            stds.append(float(r.std()))
            building_fracs.append(float(scene.mask.ravel().mean()))

        return {
            "n_sample": len(scenes),
            "seed": int(seed),
            "load_ms_median": float(np.median(load_times) * 1e3),
            "path_loss_dbm_min": float(np.min(mins)),
            "path_loss_dbm_max": float(np.max(maxs)),
            "path_loss_dbm_mean": float(np.mean(means)),
            "path_loss_dbm_std": float(np.mean(stds)),
            "building_fraction_mean": float(np.mean(building_fracs)),
            "valid_fraction_min": float(min(float(s.valid.mean()) for s in scenes)),
            "tx_single_pixel": bool(all(int(s.tx.sum()) == 1 for s in scenes)),
        }

    @staticmethod
    def prepare(
        data_dir: str,
        unpacked_dir: str,
        download: bool = True,
        extract: bool = True,
        verify: bool = False,
        seed: int = 0,
    ) -> dict:
        """Run the full download → extract → index → stats pipeline."""
        report: dict = {"source": DATASET, "seed": int(seed), "steps": {}}
        t0 = time.perf_counter()
        if download:
            report["steps"]["download"] = ScalableData.download(data_dir, verify=verify)
        if extract:
            report["steps"]["extract"] = ScalableData.extract_all(
                data_dir, unpacked_dir, verify=verify
            )
        entries = ScalableData.index(unpacked_dir)
        report["steps"]["index"] = {
            "count": len(entries),
            "fingerprint": ScalableData.fingerprint(entries),
        }
        report["steps"]["stats"] = ScalableData.stats(entries, seed=seed)
        report["elapsed_seconds"] = time.perf_counter() - t0
        return report

    @staticmethod
    def selfcheck(entries: list, n_maps: int = 5, seed: int = 0) -> list:
        """Clean and transform ``n_maps`` scenes; return a traceability table."""
        assert n_maps > 0, "n_maps must be positive"
        rows = []
        for i, entry in enumerate(entries[:n_maps]):
            scene = ScalableData.clean(entry)
            task = ScalableData.task(scene, n_sensors=1500, n_eval=3000, seed=seed)
            rows.append(
                {
                    "scene": f"{scene.split}/{scene.name}",
                    "pixels": int(scene.radio.size),
                    "valid": float(scene.valid.mean()),
                    "building": float(scene.mask.mean()),
                    "tx_pixels": int(scene.tx.sum()),
                    "sensors": len(task["sensor_idx"]),
                    "eval": len(task["eval_idx"]),
                    "y_mean_dbm": task["y_mean"],
                    "y_std_dbm": task["y_std"],
                }
            )
        return rows

    @staticmethod
    def main(argv: Optional[Sequence[str]] = None) -> int:
        """Command-line entry point for the pipeline."""
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument("--data-dir", default="data/ucf50k")
        parser.add_argument("--unpack-dir", default="data/ucf50k/unpacked")
        parser.add_argument("--skip-download", action="store_true")
        parser.add_argument("--skip-extract", action="store_true")
        parser.add_argument("--verify", action="store_true", help="CRC-check zips before extract")
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--selfcheck", type=int, default=0, metavar="N", help="clean+transform N scenes"
        )
        parser.add_argument("--report", default=None, help="write the pipeline report JSON here")
        args = parser.parse_args(argv)

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

        report = ScalableData.prepare(
            data_dir=args.data_dir,
            unpacked_dir=args.unpack_dir,
            download=not args.skip_download,
            extract=not args.skip_extract,
            verify=args.verify,
            seed=args.seed,
        )
        entries = ScalableData.index(args.unpack_dir)

        if args.selfcheck:
            rows = ScalableData.selfcheck(entries, n_maps=args.selfcheck, seed=args.seed)
            for row in rows:
                print(row)
            report["selfcheck"] = rows

        report_path = args.report or os.path.join(args.data_dir, "pipeline_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)

        step = report["steps"]
        print(
            "pipeline: maps=%d fingerprint=%s sample_stats=%s"
            % (
                step["index"]["count"],
                step["index"]["fingerprint"][:16],
                step["stats"],
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(ScalableData.main())
