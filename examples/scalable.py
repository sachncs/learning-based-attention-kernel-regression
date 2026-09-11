"""Scalable: reproduce LAKER on the real-world UCF-50K corpus.

End-to-end, fully reproducible experiment on the 50,000-map
``KR-init/Spectrum-Cartography-256x256-UCF-50K`` dataset. Runs a
quantitative full-sweep comparison of every LAKER kernel, picks the
best configuration on a validation set, then evaluates that
configuration on the complete 256×256 radio grid over the whole
corpus (resumable).

Method (staged protocol):

1. ``sweep``  — every kernel configuration × ``--max-maps`` maps from
   the ``val`` split. Per-scene metrics: RMSE, MAE, R² (dBm), fit /
   predict time, PCG iterations, condition number, and a mean-target
   baseline. Aggregates a per-configuration table, a win-rate vs
    ``exact``, and the accuracy-vs-time Pareto front.
2. ``validate`` — the winning configuration is re-fitted on every map
   (default ``--validate-maps`` maps; ``--validate-maps 0`` runs all
   50,000) and scored on the *complete* 256×256 grid masked to
   non-building pixels. Results stream to an append-only CSV so the
   full-corpus run is resumable and can be distributed with
   ``--workers``. With ``--cross-map``, each map is additionally scored
   against the *stationary cross-map prior*: the per-pixel mean map
   aggregated over the train split (cached under
   ``data/ucf50k/cross_map_mean.npy`` after first build).

Reproducibility: every run writes, under ``outputs/scalable/<run>/``,
an event log (JSONL), the sweep / validation CSVs, a metrics summary,
a run manifest (git commit, environment fingerprint, exact
configuration, seeds) and the dataset fingerprint. Scene sampling is
deterministic for a given ``--seed``; targets are standardized per
scene with the sensor-batch statistics and predictions un-standardized
so a constant predictor scores zero.

Run::

    python -m examples.scalable --max-maps 25 --validate-maps 25
    python -m examples.scalable --validate-maps 0 --workers 8

The second command evaluates the winning configuration on the entire
50,000-map corpus.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Sequence

import numpy as np
import torch

from examples.scalable_data import EXPECTED_COUNT, SIDE, ScalableData
from laker import Laker
from laker.backend import Backend

logger = logging.getLogger(__name__)

LAMS = [1e-2, 1e-1]
BASE: dict = {
    "embed_dim": 16,
    "gamma": 1e-1,
    "num": 50,
    "cccp_max": 50,
    "cccp_tol": 1e-6,
    "pcg_tol": 1e-6,
    "pcg_max": 500,
}
SWEEP: list[tuple[str, dict]] = [
    ("exact", {"kernel_type": "exact"}),
    ("nystrom_m100", {"kernel_type": "nystrom", "landmarks": 100}),
    ("nystrom_m256", {"kernel_type": "nystrom", "landmarks": 256}),
    ("nystrom_m512", {"kernel_type": "nystrom", "landmarks": 512}),
    ("fourier_r128", {"kernel_type": "fourier", "features": 128}),
    ("fourier_r256", {"kernel_type": "fourier", "features": 256}),
    ("fourier_r512", {"kernel_type": "fourier", "features": 512}),
    ("fourier_r1024", {"kernel_type": "fourier", "features": 1024}),
    ("neighbors_k5", {"kernel_type": "neighbors", "neighbors": 5}),
    ("neighbors_k10", {"kernel_type": "neighbors", "neighbors": 10}),
    ("neighbors_k20", {"kernel_type": "neighbors", "neighbors": 20}),
    ("grid_g64", {"kernel_type": "grid", "grid_size": 64, "embed_dim": 2}),
    ("spectrum_k5", {"kernel_type": "spectrum", "knots": 5}),
    ("hybrid_b0.3", {"kernel_type": "hybrid", "landmarks": 128, "neighbors": 10, "blend": 0.3}),
    ("hybrid_b0.5", {"kernel_type": "hybrid", "landmarks": 128, "neighbors": 10, "blend": 0.5}),
    ("hybrid_b0.7", {"kernel_type": "hybrid", "landmarks": 128, "neighbors": 10, "blend": 0.7}),
]
SWEEP_COLUMNS = [
    "scene", "split", "config", "kernel", "lam", "params", "status", "rmse", "mae", "r2",
    "baseline_rmse", "fit_s", "predict_s", "total_s", "iters", "condition", "scene_seed",
]
VALIDATE_COLUMNS = [
    "scene", "split", "sensors", "eval_pixels", "coverage", "rmse", "mae", "r2",
    "baseline_rmse", "cross_map_rmse", "fit_s", "predict_s", "iters", "scene_seed",
]


def _cross_map_accumulate(chunk: list) -> tuple[np.ndarray, np.ndarray]:
    """Per-chunk accumulator for the train-split mean map (module-level for pickling)."""
    acc: np.ndarray = np.zeros((SIDE, SIDE), dtype=np.float64)
    cnt: np.ndarray = np.zeros((SIDE, SIDE), dtype=np.float64)
    for entry in chunk:
        scene = ScalableData.clean(entry)
        mask = scene.valid
        acc[mask] += scene.radio[mask]
        cnt[mask] += 1
    return acc, cnt


class Scalable:
    """Reproduce LAKER on the UCF-50K corpus with a full kernel sweep."""

    @staticmethod
    def provenance() -> dict:
        """Capture a reproducible environment fingerprint."""
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.getcwd()
        )
        git_dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=os.getcwd()
        )
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_mps_available": bool(
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ),
            "num_cpu": os.cpu_count(),
            "threads": torch.get_num_threads(),
            "git_commit": git.stdout.strip() if git.returncode == 0 else "n/a",
            "git_dirty": bool(git_dirty.stdout.strip()),
        }

    @staticmethod
    def build_configs(
        lams: Optional[Sequence[float]] = None,
        groups: Optional[Sequence[str]] = None,
    ) -> list:
        """Expand the sweep into one config dict per (kernel, lambda)."""
        lams = list(lams if lams is not None else LAMS)
        configs = []
        for group, kwargs in SWEEP:
            if groups is not None and group not in groups:
                continue
            for lam in lams:
                configs.append(
                    {
                        "name": f"{group}_lam{lam:g}",
                        "kernel": group,
                        "lam": float(lam),
                        "kwargs": dict(kwargs),
                    }
                )
        return configs

    @staticmethod
    def scene_seed(seed: int, split: str, name: str) -> int:
        """Derive a stable per-scene RNG seed from the global seed."""
        digest = hashlib.sha256(f"{seed}:{split}:{name}".encode()).digest()[:8]
        return int.from_bytes(digest, "little")

    @staticmethod
    def select(entries: list, split: str, max_maps: int, seed: int) -> list:
        """Deterministically select ``max_maps`` maps from ``split``."""
        pool = [entry for entry in entries if entry["split"] == split]
        assert max_maps <= len(pool), (
            f"max_maps={max_maps} exceeds {split} maps available ({len(pool)})"
        )
        order = sorted(pool, key=lambda entry: Scalable.scene_seed(seed, split, entry["name"]))
        return order[:max_maps]

    @staticmethod
    def baseline(task: dict) -> dict:
        """RMSE/MAE/R² of predicting the eval set by the sensor mean."""
        err = task["ye_raw"] - task["y_mean"]
        ss_tot = float(np.sum((task["ye_raw"] - task["ye_raw"].mean()) ** 2))
        ss_res = float(np.sum(err**2))
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        }

    @staticmethod
    def metrics(task: dict, pred_dbm: np.ndarray) -> dict:
        """RMSE/MAE/R² of ``pred_dbm`` against the eval targets (dBm)."""
        err = pred_dbm - task["ye_raw"]
        ss_tot = float(np.sum((task["ye_raw"] - task["ye_raw"].mean()) ** 2))
        ss_res = float(np.sum(err**2))
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        }

    @staticmethod
    def run_config(
        task: dict,
        cfg: dict,
        device: str = "cpu",
        seed: int = 0,
        condition: bool = True,
        dtype: str = "float64",
    ) -> dict:
        """Fit and evaluate one configuration on one scene's task.

        Returns a metric row, or a ``status="breakdown"`` row with the
        exception message when the configuration is numerically unstable
        (recorded, never silently dropped).
        """
        kwargs: dict = dict(BASE)
        kwargs.update(cfg["kwargs"])
        kwargs.update(
            {
                "lam": cfg["lam"],
                "device": device,
                "dtype": torch.float64 if dtype == "float64" else torch.float32,
                "verbose": False,
            }
        )
        safe = {
            key: (str(value) if isinstance(value, (torch.dtype, torch.device)) else value)
            for key, value in kwargs.items()
        }
        row = {
            "config": cfg["name"],
            "kernel": cfg["kernel"],
            "lam": cfg["lam"],
            "params": json.dumps(safe, sort_keys=True),
            "status": "ok",
            "rmse": None, "mae": None, "r2": None, "baseline_rmse": None,
            "fit_s": None, "predict_s": None, "total_s": None,
            "iters": None, "condition": None,
            "scene_seed": int(seed),
        }
        try:
            model = Laker(**kwargs)
            torch.manual_seed(seed)
            t0 = time.perf_counter()
            model.fit(torch.as_tensor(task["x"]), torch.as_tensor(task["y"]))
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred = model.predict(torch.as_tensor(task["xe"]))
            predict_s = time.perf_counter() - t0
            pred_dbm = (task["y_std"] * pred.detach().cpu().numpy()) + task["y_mean"]

            assert np.isfinite(pred_dbm).all(), f"{cfg['name']}: non-finite predictions"
            score = Scalable.metrics(task, pred_dbm)
            baseline_rmse = Scalable.baseline(task)["rmse"]
            if score["rmse"] > 10 * baseline_rmse:
                row["status"] = "degenerate"
                row["params"] = (
                    f"{row['params']} | rmse {score['rmse']:.2e} > 10x "
                    f"baseline {baseline_rmse:.2f}"
                )
            row.update(score)
            row["baseline_rmse"] = baseline_rmse
            row["fit_s"] = fit_s
            row["predict_s"] = predict_s
            row["total_s"] = fit_s + predict_s
            row["iters"] = int(model.iters)
            if condition:
                try:
                    row["condition"] = float(model.condition())
                except Exception as exc:  # pragma: no cover - defensive
                    row["condition"] = f"error:{type(exc).__name__}"
        except Exception as exc:
            row["status"] = "breakdown"
            row["params"] = f"{row['params']} | error={type(exc).__name__}: {exc}"
        return row

    @staticmethod
    def aggregate(rows: list) -> list:
        """Collapse per-scene rows into one summary row per configuration."""
        exact = {
            r["scene"]: r["rmse"]
            for r in rows
            if r["status"] == "ok" and r["kernel"] == "exact"
        }
        exact_total = None
        exact_rows = [r for r in rows if r["status"] == "ok" and r["kernel"] == "exact"]
        if exact_rows:
            exact_total = max(r["total_s"] for r in exact_rows)

        by_config: dict = {}
        for r in rows:
            by_config.setdefault(r["config"], []).append(r)

        summary = []
        for config, group in sorted(by_config.items()):
            ok = [r for r in group if r["status"] == "ok"]
            first = group[0]
            if not ok:
                summary.append(
                    {
                        "config": config,
                        "kernel": first["kernel"],
                        "lam": first["lam"],
                        "n": len(group),
                        "n_ok": 0,
                        "status": first["status"],
                    }
                )
                continue
            rmses = np.array([r["rmse"] for r in ok])
            totals = np.array([r["total_s"] for r in ok])
            wins = sum(1 for r in ok if r["scene"] in exact and r["rmse"] < exact[r["scene"]])
            summary.append(
                {
                    "config": config,
                    "kernel": first["kernel"],
                    "lam": first["lam"],
                    "n": len(group),
                    "n_ok": len(ok),
                    "status": "ok",
                    "mean_rmse": float(rmses.mean()),
                    "median_rmse": float(np.median(rmses)),
                    "mean_mae": float(np.mean([r["mae"] for r in ok])),
                    "mean_r2": float(np.mean([r["r2"] for r in ok])),
                    "mean_fit_s": float(np.mean([r["fit_s"] for r in ok])),
                    "mean_predict_s": float(np.mean([r["predict_s"] for r in ok])),
                    "mean_total_s": float(totals.mean()),
                    "mean_iters": float(np.mean([r["iters"] for r in ok])),
                    "speedup": (exact_total / totals.mean()) if exact_total else None,
                    "win_rate_vs_exact": wins / max(1, len(ok)),
                }
            )
        return sorted(summary, key=lambda s: s.get("mean_rmse", float("inf")))

    @staticmethod
    def pareto(summary: list) -> list:
        """Accuracy-vs-time Pareto front (minimising RMSE and time)."""
        ok = [s for s in summary if s["status"] == "ok"]
        ordered = sorted(ok, key=lambda s: (s["mean_rmse"], s["mean_total_s"]))
        front = []
        best_time = float("inf")
        for entry in ordered:
            if entry["mean_total_s"] < best_time:
                front.append(
                    {
                        "config": entry["config"],
                        "mean_rmse": entry["mean_rmse"],
                        "mean_total_s": entry["mean_total_s"],
                    }
                )
                best_time = entry["mean_total_s"]
        return front

    @staticmethod
    def validate_scene(
        entry: dict,
        cfg: dict,
        sensors: int = 2000,
        seed: int = 0,
        device: str = "cpu",
        dtype: str = "float64",
        cross_map: Optional[np.ndarray] = None,
    ) -> dict:
        """Fit ``cfg`` on a scene and score the complete masked 256×256 grid."""
        scene = ScalableData.clean(entry)
        scene_seed = Scalable.scene_seed(seed, scene.split, scene.name)
        task = ScalableData.task(scene, n_sensors=sensors, n_eval=5000, seed=scene_seed)
        baseline = Scalable.baseline(task)

        kwargs: dict = dict(BASE)
        kwargs.update(cfg["kwargs"])
        kwargs.update(
            {
                "lam": cfg["lam"],
                "device": device,
                "dtype": torch.float64 if dtype == "float64" else torch.float32,
                "verbose": False,
            }
        )
        model = Laker(**kwargs)
        torch.manual_seed(scene_seed)
        t0 = time.perf_counter()
        model.fit(torch.as_tensor(task["x"]), torch.as_tensor(task["y"]))
        fit_s = time.perf_counter() - t0

        grid = ScalableData.grid()
        t0 = time.perf_counter()
        full = model.predict(torch.as_tensor(grid))
        predict_s = time.perf_counter() - t0
        full_dbm = (task["y_std"] * full.detach().cpu().numpy()) + task["y_mean"]
        assert np.isfinite(full_dbm).all(), f"{scene.name}: non-finite full-grid predictions"

        valid_idx = np.flatnonzero(scene.valid)
        truth = scene.radio.ravel()[valid_idx]
        pred = full_dbm[valid_idx]
        err = pred - truth
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        ss_res = float(np.sum(err**2))
        cross_map_rmse = (
            Scalable.cross_map_rmse(cross_map, scene) if cross_map is not None else None
        )
        return {
            "scene": f"{scene.split}/{scene.name}",
            "split": scene.split,
            "sensors": int(sensors),
            "eval_pixels": int(valid_idx.size),
            "coverage": float(valid_idx.size / scene.radio.size),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            "baseline_rmse": baseline["rmse"],
            "cross_map_rmse": cross_map_rmse,
            "fit_s": fit_s,
            "predict_s": predict_s,
            "iters": int(model.iters),
            "scene_seed": int(scene_seed),
        }

    @staticmethod
    def cross_map_mean(
        entries: list, workers: int, data_dir: str, events
    ) -> np.ndarray:
        """Build (or load) the per-pixel mean map aggregated over the train split.

        The mean map is the stationary cross-map prior: every held-out scene
        is scored against this single fixed prediction. Cached under
        ``<data_dir>/cross_map_mean.npy``; subsequent runs reuse it. Valid
        (non-building) pixels carry the empirical mean, building pixels are
        left as NaN.
        """
        cache = os.path.join(data_dir, "cross_map_mean.npy")
        if os.path.exists(cache):
            mean = np.load(cache)
            assert mean.shape == (SIDE, SIDE), f"corrupt cross-map cache: {mean.shape}"
            Scalable.event(events, "cross_map_cached", {"path": cache, "shape": list(mean.shape)})
            return mean
        train = [entry for entry in entries if entry["split"] == "train"]
        Scalable.event(
            events, "cross_map_build_start", {"train_maps": len(train), "workers": workers}
        )

        if workers and workers > 1:
            chunks = [train[i::workers] for i in range(workers)]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_cross_map_accumulate, chunks))
        else:
            results = [_cross_map_accumulate(train)]
        acc_agg: np.ndarray = np.zeros((SIDE, SIDE), dtype=np.float64)
        cnt_agg: np.ndarray = np.zeros((SIDE, SIDE), dtype=np.float64)
        for partial_acc, partial_cnt in results:
            acc_agg += partial_acc
            cnt_agg += partial_cnt
        mean = acc_agg / np.maximum(cnt_agg, 1.0)
        mean[cnt_agg == 0] = np.nan
        np.save(cache, mean)
        Scalable.event(
            events,
            "cross_map_built",
            {
                "path": cache,
                "train_maps": len(train),
                "covered_pixels": int(np.count_nonzero(cnt_agg)),
                "mean_of_means": float(np.nanmean(mean)),
            },
        )
        return mean

    @staticmethod
    def cross_map_rmse(mean_map: np.ndarray, scene) -> Optional[float]:
        """RMSE of the cached mean map against ``scene.radio`` over valid pixels.

        Returns ``None`` if the mean map has no coverage for this scene.
        """
        valid_idx = np.flatnonzero(scene.valid)
        pred = np.asarray(mean_map).ravel()[valid_idx]
        truth = scene.radio.ravel()[valid_idx]
        good = np.isfinite(pred)
        if not good.any():
            return None
        err = pred[good] - truth[good]
        return float(np.sqrt(np.mean(err**2)))

    @staticmethod
    def event(handle, event: str, payload: dict) -> None:
        """Append one timestamped event to the JSONL observability log."""
        handle.write(
            json.dumps(
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "event": event,
                    **payload,
                }
            )
            + "\n"
        )
        handle.flush()

    @staticmethod
    def write_rows(path: str, columns: list, rows: list) -> None:
        """Atomically (re)write a CSV of ``rows``."""
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp, path)

    @staticmethod
    def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("--data-dir", default="data/ucf50k")
        parser.add_argument("--unpack-dir", default="data/ucf50k/unpacked")
        parser.add_argument("--out-dir", default="outputs/scalable")
        parser.add_argument("--skip-download", action="store_true")
        parser.add_argument("--skip-extract", action="store_true")
        parser.add_argument("--max-maps", type=int, default=25)
        parser.add_argument("--sweep-split", default="val")
        parser.add_argument("--sensors", type=int, default=2000)
        parser.add_argument("--eval-points", type=int, default=8000)
        parser.add_argument("--lams", type=float, nargs="+", default=LAMS)
        parser.add_argument(
            "--configs", default=None, help="comma-separated kernel groups to sweep"
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--device", default="cpu")
        parser.add_argument(
            "--dtype",
            choices=("float64", "float32"),
            default="float64",
            help="solver precision (float32 ~2x faster, slight accuracy change)",
        )
        parser.add_argument("--condition", action="store_true", help="estimate kappa per config")
        parser.add_argument(
            "--validate-maps",
            type=int,
            default=None,
            metavar="N",
            help="maps to validate (default = max-maps; 0 = ALL maps in validate-splits)",
        )
        parser.add_argument(
            "--validate-splits", default="test,val,train"
        )
        parser.add_argument(
            "--cross-map",
            action="store_true",
            help=(
                "score every map against the per-pixel mean map aggregated over the "
                "train split (cached under <data-dir>/cross_map_mean.npy); a stationary "
                "cross-map prior that bounds naive cross-map learning without per-scene "
                "conditioning. Computes the cache once (~5 min on first run, then reused)."
            ),
        )
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument(
            "--resume",
            default=None,
            metavar="RUN_ID",
            help="resume an existing run (reuse its output dir and validation checkpoint)",
        )
        parser.add_argument("--verbose", action="store_true")
        return parser.parse_args(argv)

    @staticmethod
    def main(argv: Optional[Sequence[str]] = None) -> int:
        args = Scalable.parse_args(argv)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
        Backend.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.validate_maps is None:
            args.validate_maps = args.max_maps

        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if args.resume:
            run_id = args.resume
            run_dir = os.path.join(args.out_dir, args.resume)
            assert os.path.isdir(run_dir), (
                f"run {args.resume} not found under {args.out_dir}; "
                "resume must reuse the same seed, splits, sensors and validation size"
            )
        else:
            run_dir = os.path.join(args.out_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        events_path = os.path.join(run_dir, "events.jsonl")
        events = open(events_path, "a", encoding="utf-8")

        groups = [g.strip() for g in args.configs.split(",")] if args.configs else None
        configs = Scalable.build_configs(lams=args.lams, groups=groups)
        provenance = Scalable.provenance()
        Scalable.event(
            events,
            "start",
            {
                "run_id": run_id,
                "argv": list(sys.argv),
                "configs": len(configs),
                "provenance": provenance,
            },
        )

        report = ScalableData.prepare(
            data_dir=args.data_dir,
            unpacked_dir=args.unpack_dir,
            download=not args.skip_download,
            extract=not args.skip_extract,
            verify=False,
            seed=args.seed,
        )
        entries = ScalableData.index(args.unpack_dir)
        fingerprint = ScalableData.fingerprint(entries)
        assert len(entries) == sum(EXPECTED_COUNT.values()), (
            f"corpus index: expected {sum(EXPECTED_COUNT.values())} maps, got {len(entries)}"
        )
        assert fingerprint == report["steps"]["index"]["fingerprint"], "corpus fingerprint drift"
        Scalable.event(
            events,
            "data_ready",
            {
                "maps": len(entries),
                "fingerprint": fingerprint,
                "stats": report["steps"]["stats"],
            },
        )

        scenes = Scalable.select(entries, args.sweep_split, args.max_maps, args.seed)
        assert len(scenes) == args.max_maps, "scene selection mismatch"
        Scalable.event(
            events,
            "sweep_scenes",
            {
                "split": args.sweep_split,
                "count": len(scenes),
                "scenes": [f"{s['split']}/{s['name']}" for s in scenes],
            },
        )

        rows = []
        for scene in scenes:
            clean = ScalableData.clean(scene)
            scene_seed = Scalable.scene_seed(args.seed, clean.split, clean.name)
            task = ScalableData.task(
                clean, n_sensors=args.sensors, n_eval=args.eval_points, seed=scene_seed
            )
            for cfg in configs:
                row = Scalable.run_config(
                    task, cfg, device=args.device, seed=scene_seed, condition=args.condition,
                    dtype=args.dtype,
                )
                row.update(
                    {
                        "scene": f"{clean.split}/{clean.name}",
                        "split": clean.split,
                        "lam": cfg["lam"],
                    }
                )
                rows.append(row)
                Scalable.event(
                    events,
                    "sweep_config_done",
                    {
                        "scene": row["scene"],
                        "config": row["config"],
                        "status": row["status"],
                        "rmse": row["rmse"],
                        "total_s": row["total_s"],
                    },
                )

        sweep_csv = os.path.join(run_dir, "sweep.csv")
        Scalable.write_rows(sweep_csv, SWEEP_COLUMNS, rows)
        summary = Scalable.aggregate(rows)
        front = Scalable.pareto(summary)
        ok_summary = [s for s in summary if s["status"] == "ok"]
        winner = ok_summary[0] if ok_summary else None
        assert winner is not None, "no configuration produced a valid fit"
        baseline_ok = [r["baseline_rmse"] for r in rows if r["status"] == "ok"]
        baseline_mean = float(np.mean(baseline_ok))

        assert winner["mean_rmse"] < baseline_mean * 0.65, (
            f"winner {winner['config']} RMSE {winner['mean_rmse']:.2f} dB does not beat the "
            f"baseline {baseline_mean:.2f} dB by >35%"
        )
        exact_row = next((s for s in ok_summary if s["kernel"] == "exact"), None)
        if exact_row is not None:
            assert winner["mean_rmse"] <= exact_row["mean_rmse"] * 1.05, (
                f"winner {winner['config']} RMSE {winner['mean_rmse']:.2f} exceeds exact "
                f"{exact_row['mean_rmse']:.2f} by >5%"
            )
        Scalable.event(
            events,
            "sweep_done",
            {
                "n_scenes": len(scenes),
                "n_rows": len(rows),
                "baseline_mean_rmse": baseline_mean,
                "winner": winner,
                "pareto_front": front,
            },
        )

        validate_rows = Scalable.validate(
            entries, configs, winner, args, run_dir, events, report
        )

        metrics = {
            "dataset": {
                "source": ScalableData.source,
                "revision": ScalableData.revision,
                "data_version": getattr(ScalableData, "revision", "main"),
                "maps": len(entries),
                "fingerprint": fingerprint,
            },
            "config": {
                "lams": list(args.lams),
                "base": BASE,
                "sweep_split": args.sweep_split,
                "max_maps": args.max_maps,
                "sensors": args.sensors,
                "eval_points": args.eval_points,
                "seed": args.seed,
                "device": args.device,
                "dtype": args.dtype,
                "cross_map": bool(args.cross_map),
            },
            "sweep": {
                "n_configs": len(configs),
                "n_scenes": len(scenes),
                "baseline_mean_rmse": baseline_mean,
                "winner": winner,
                "table": summary,
                "pareto_front": front,
            },
            "validate": {"count": len(validate_rows), "rows": validate_rows},
        }
        if args.cross_map:
            cross_map_rows = [
                row for row in validate_rows
                if row.get("cross_map_rmse") not in (None, "")
            ]
            if cross_map_rows:
                per_split: dict = {}
                for split in sorted({row["split"] for row in cross_map_rows}):
                    rmses = [
                        float(row["cross_map_rmse"])
                        for row in cross_map_rows
                        if row["split"] == split
                    ]
                    per_split[split] = {
                        "n": len(rmses),
                        "mean_rmse": float(np.mean(rmses)),
                        "median_rmse": float(np.median(rmses)),
                    }
                metrics["cross_map"] = {
                    "baseline": (
                        "per-pixel mean map over the train split (stationary "
                        "cross-map prior); cached at <data-dir>/cross_map_mean.npy"
                    ),
                    "cache": os.path.join(args.data_dir, "cross_map_mean.npy"),
                    "n_scenes": len(cross_map_rows),
                    "overall_mean_rmse": float(np.mean(
                        [float(row["cross_map_rmse"]) for row in cross_map_rows]
                    )),
                    "per_split": per_split,
                }
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        manifest = {
            "run_id": run_id,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "argv": list(sys.argv),
            "provenance": provenance,
            "dataset": metrics["dataset"],
            "config": metrics["config"],
            "winner": winner,
            "files": {
                "events": events_path,
                "sweep_csv": sweep_csv,
                "metrics": metrics_path,
                "validate_csv": os.path.join(run_dir, "validate.csv"),
            },
        }
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        events.close()

        Scalable.print_summary(summary, front, winner, baseline_mean, validate_rows)
        print(f"run artifacts: {run_dir}")
        return 0

    @staticmethod
    def validate(
        entries: list,
        configs: list,
        winner: dict,
        args: argparse.Namespace,
        run_dir: str,
        events,
        report: dict,
    ) -> list:
        """Score the winner on a deterministic subset (or all) of the corpus."""
        want = {
            entry["split"]
            for entry in entries
            if entry["split"] in args.validate_splits.split(",")
        }
        pool = [entry for entry in entries if entry["split"] in want]
        if args.validate_maps and args.validate_maps > 0:
            stride = max(1, len(pool) // args.validate_maps)
            pool = [pool[i] for i in range(0, len(pool), stride)][: args.validate_maps]
        pool = sorted(
            pool, key=lambda entry: Scalable.scene_seed(args.seed, entry["split"], entry["name"])
        )
        winner_cfg = next(cfg for cfg in configs if cfg["name"] == winner["config"])

        csv_path = os.path.join(run_dir, "validate.csv")
        existing: set = set()
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as handle:
                existing = {row["scene"] for row in csv.DictReader(handle)}
        remaining = [entry for entry in pool if f"{entry['split']}/{entry['name']}" not in existing]

        cross_map: Optional[np.ndarray] = None
        if args.cross_map:
            cross_map = Scalable.cross_map_mean(entries, args.workers, args.data_dir, events)

        Scalable.event(
            events,
            "validate_start",
            {
                "config": winner_cfg["name"],
                "splits": sorted(want),
                "requested": len(pool),
                "remaining": len(remaining),
                "workers": args.workers,
                "cross_map": cross_map is not None,
            },
        )

        def record(row: dict) -> None:
            Scalable.event(
                events,
                "validate_scene_done",
                {"scene": row["scene"], "rmse": row["rmse"], "coverage": row["coverage"]},
            )
            write_header = not existing and not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=VALIDATE_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        if args.workers and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
                futures = {
                    pool_exec.submit(
                        Scalable.validate_scene,
                        entry,
                        winner_cfg,
                        args.sensors,
                        args.seed,
                        args.device,
                        args.dtype,
                        cross_map,
                    ): entry
                    for entry in remaining
                }
                for future in as_completed(futures):
                    record(future.result())
        else:
            for entry in remaining:
                record(
                    Scalable.validate_scene(
                        entry,
                        winner_cfg,
                        sensors=args.sensors,
                        seed=args.seed,
                        device=args.device,
                        dtype=args.dtype,
                        cross_map=cross_map,
                    )
                )

        all_rows = []
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as handle:
                all_rows = list(csv.DictReader(handle))
        Scalable.event(
            events,
            "validate_done",
            {"completed": len(all_rows), "added_this_run": len(all_rows) - len(existing)},
        )
        return all_rows

    @staticmethod
    def print_summary(
        summary: list, front: list, winner: dict, baseline_mean: float, validate_rows: list
    ) -> None:
        """Render the human-readable console report."""
        print("\n=== kernel sweep (UCF-50K, real ray-traced maps) ===")
        print(
            f"{'config':<20} {'status':<10} {'rmse':>7} {'mae':>7} {'r2':>6} {'fit_s':>7} "
            f"{'pred_s':>7} {'iters':>6} {'speedup':>8}"
        )
        for s in sorted(summary, key=lambda e: e.get("mean_rmse", float("inf"))):
            if s["status"] != "ok":
                print(f"{s['config']:<20} {s['status']:<10}")
                continue
            speedup = f"{s['speedup']:.2f}x" if s["speedup"] else "-"
            print(
                f"{s['config']:<20} {'ok':<10} {s['mean_rmse']:7.2f} {s['mean_mae']:7.2f} "
                f"{s['mean_r2']:6.3f} {s['mean_fit_s']:7.3f} {s['mean_predict_s']:7.3f} "
                f"{s['mean_iters']:6.0f} {speedup:>8}"
            )
        print(f"\nbaseline (mean-target) RMSE: {baseline_mean:.2f} dB")
        print(f"winner: {winner['config']}  RMSE {winner['mean_rmse']:.2f} dB  "
              f"({100 * (1 - winner['mean_rmse'] / baseline_mean):.1f}% below baseline)")
        print("pareto front (accuracy vs time):")
        for point in front:
            print(
                f"  {point['config']:<20} rmse={point['mean_rmse']:6.2f} "
                f"total={point['mean_total_s']:6.3f}s"
            )
        if validate_rows:
            rmses = [float(r["rmse"]) for r in validate_rows]
            covers = [float(r["coverage"]) for r in validate_rows]
            base = [float(r["baseline_rmse"]) for r in validate_rows]
            print(
                f"\n=== full-grid validation ({len(validate_rows)} maps) ==="
            )
            print(f"masked RMSE: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f} dB  "
                  f"(baseline {np.mean(base):.2f} dB)")
            print(f"coverage (valid pixels): {np.mean(covers):.2%}  |  "
                  f"{np.min(rmses):.2f} .. {np.max(rmses):.2f} dB range")
            cross_map_rmses = [
                float(r["cross_map_rmse"]) for r in validate_rows
                if r.get("cross_map_rmse") not in (None, "")
            ]
            if cross_map_rmses:
                print(
                    f"cross-map (per-pixel train mean) RMSE: "
                    f"{np.mean(cross_map_rmses):.2f} +/- {np.std(cross_map_rmses):.2f} dB  "
                    f"on {len(cross_map_rmses)} maps"
                )


if __name__ == "__main__":
    raise SystemExit(Scalable.main())
