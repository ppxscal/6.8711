"""Candidate scoring with RTMScore."""
from __future__ import annotations

import dataclasses
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm.auto import tqdm

from chorus.config import Config
from chorus.generators import ligand_properties
from chorus.pockets import PocketSpec
from chorus.runtime import discover_devices, env_python, run_command, visible_gpu_env

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# RTMScore pose scorer
# ---------------------------------------------------------------------------

@dataclass
class PoseRecord:
    pose_id: str
    smiles: str
    generator: str
    pocket_id: str
    source_rank: int
    pose_sdf: str
    source: str


def canonical_smiles_from_mol(mol: Chem.Mol) -> str | None:
    try:
        clean = Chem.Mol(mol)
        try:
            clean = Chem.RemoveHs(clean)
        except Exception:
            pass
        return Chem.MolToSmiles(clean) or None
    except Exception:
        return None


def first_mol_from_sdf(path: Path) -> Chem.Mol | None:
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        for mol in supplier:
            if mol is not None:
                return mol
    except Exception:
        return None
    return None


def pose_id_for(*parts: object) -> str:
    key = "|".join(str(p) for p in parts)
    return f"pose_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def allowed_pose_keys(generated_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if generated_df.empty:
        return set()
    return {
        (str(r.generator), str(r.pocket_id), str(r.smiles))
        for r in generated_df[["generator", "pocket_id", "smiles"]].itertuples(index=False)
    }


def latest_gen_info_by_pocket(generator_root: Path) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for csv_path in generator_root.rglob("gen_info.csv"):
        try:
            pocket_id = csv_path.relative_to(generator_root).parts[0]
        except Exception:
            continue
        current = latest.get(pocket_id)
        if current is None or csv_path.stat().st_mtime >= current.stat().st_mtime:
            latest[pocket_id] = csv_path
    return latest


def find_pocketxmol_pose_sdf(csv_path: Path, filename: str) -> Path | None:
    filename = filename.strip()
    if not filename:
        return None
    candidates = [
        csv_path.parent / f"{csv_path.parent.name}_SDF" / filename,
        csv_path.parent / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(csv_path.parent.rglob(filename))
    return matches[0] if matches else None


def collect_pocketxmol_pose_records(
    run_dir: Path,
    generated_df: pd.DataFrame,
    generator_name: str,
    allowed: set[tuple[str, str, str]],
) -> list[tuple[PoseRecord, Chem.Mol]]:
    rows: list[tuple[PoseRecord, Chem.Mol]] = []
    root = run_dir / "generator_outputs" / generator_name.lower()
    if not root.exists():
        return rows

    for pocket_id, csv_path in sorted(latest_gen_info_by_pocket(root).items()):
        try:
            info = pd.read_csv(csv_path)
        except Exception:
            continue
        if "filename" not in info.columns or "smiles" not in info.columns:
            continue

        for row_idx, row in info.iterrows():
            filename = str(row.get("filename") or "")
            if not filename or filename.lower() == "nan":
                continue
            sdf_path = find_pocketxmol_pose_sdf(csv_path, filename)
            if sdf_path is None or "-bad" in sdf_path.stem:
                continue

            mol = first_mol_from_sdf(sdf_path)
            if mol is None:
                continue
            smi = str(row.get("smiles") or "").strip()
            if not smi or not ligand_properties(smi):
                smi = canonical_smiles_from_mol(mol) or ""
            if not smi or not ligand_properties(smi):
                continue
            if (generator_name, pocket_id, smi) not in allowed:
                continue

            pose_id = pose_id_for(generator_name, pocket_id, csv_path.parent.name, filename, row_idx)
            rows.append((
                PoseRecord(
                    pose_id=pose_id,
                    smiles=smi,
                    generator=generator_name,
                    pocket_id=pocket_id,
                    source_rank=int(row_idx) + 1,
                    pose_sdf=str(sdf_path),
                    source=str(csv_path),
                ),
                mol,
            ))
    return rows


def collect_diffsbdd_pose_records(
    run_dir: Path,
    generated_df: pd.DataFrame,
    allowed: set[tuple[str, str, str]],
    generator_name: str = "DiffSBDD",
) -> list[tuple[PoseRecord, Chem.Mol]]:
    rows: list[tuple[PoseRecord, Chem.Mol]] = []
    root = run_dir / "generator_outputs" / generator_name.lower()
    if not root.exists():
        return rows

    for sdf_path in sorted(root.glob("*/diffsbdd_samples.sdf")):
        pocket_id = sdf_path.parent.name
        try:
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        except Exception:
            continue
        for mol_idx, mol in enumerate(supplier):
            if mol is None:
                continue
            smi = canonical_smiles_from_mol(mol) or ""
            if not smi or not ligand_properties(smi):
                continue
            if (generator_name, pocket_id, smi) not in allowed:
                continue
            pose_id = pose_id_for(generator_name, pocket_id, sdf_path, mol_idx)
            rows.append((
                PoseRecord(
                    pose_id=pose_id,
                    smiles=smi,
                    generator=generator_name,
                    pocket_id=pocket_id,
                    source_rank=mol_idx + 1,
                    pose_sdf=str(sdf_path),
                    source=str(sdf_path),
                ),
                mol,
            ))
    return rows


def build_rtmscore_pose_batches(
    run_dir: Path,
    generated_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Build one RTMScore SDF batch per pocket from saved generator 3D poses."""
    allowed = allowed_pose_keys(generated_df)
    pose_mols: list[tuple[PoseRecord, Chem.Mol]] = []
    pose_mols.extend(collect_diffsbdd_pose_records(run_dir, generated_df, allowed, "DiffSBDD"))
    pose_mols.extend(collect_diffsbdd_pose_records(run_dir, generated_df, allowed, "DiffSBDDJoint"))
    pose_mols.extend(collect_pocketxmol_pose_records(run_dir, generated_df, "PocketXMol", allowed))
    pose_mols.extend(collect_pocketxmol_pose_records(run_dir, generated_df, "PocketXMolAR", allowed))

    seen: set[str] = set()
    deduped: list[tuple[PoseRecord, Chem.Mol]] = []
    for record, mol in pose_mols:
        if record.pose_id in seen:
            continue
        seen.add(record.pose_id)
        deduped.append((record, mol))

    score_dir = run_dir / "rtmscore"
    input_dir = score_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    sdf_by_pocket: dict[str, Path] = {}
    writers: dict[str, Chem.SDWriter] = {}
    try:
        for record, mol in deduped:
            out_sdf = input_dir / f"{record.pocket_id}_poses.sdf"
            if record.pocket_id not in writers:
                writers[record.pocket_id] = Chem.SDWriter(str(out_sdf))
                sdf_by_pocket[record.pocket_id] = out_sdf
            out_mol = Chem.Mol(mol)
            out_mol.SetProp("_Name", record.pose_id)
            out_mol.SetProp("smiles", record.smiles)
            out_mol.SetProp("generator", record.generator)
            out_mol.SetProp("pocket_id", record.pocket_id)
            writers[record.pocket_id].write(out_mol)
            manifest_rows.append(dataclasses.asdict(record))
    finally:
        for writer in writers.values():
            writer.close()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = score_dir / "rtmscore_pose_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest, sdf_by_pocket


class RTMScorePoseOracle:
    """Pose-based RTMScore scorer for generated SDF conformations."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self.repo = cfg.paths.models_dir / "rtmscore"
        self.python = env_python("rtmscore", cfg.paths)
        self.script = self.repo / "example" / "rtmscore.py"
        if not self.script.exists():
            self.script = self.repo / "rtmscore.py"
        self.model = self.repo / "trained_models" / cfg.rtmscore_model_name

    def ready(self) -> bool:
        return self.repo.exists() and self.python.exists() and self.script.exists() and self.model.exists()

    def requirements(self) -> list[str]:
        return [str(p) for p in (self.repo, self.python, self.script, self.model) if not p.exists()]

    def get_devices(self) -> list[str]:
        devices = discover_devices()
        if devices != ["cpu"]:
            return devices
        if torch is not None and torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            return devices if devices else ["cpu"]
        return ["cpu"]

    def _score_pocket(
        self,
        pocket_id: str,
        pocket_pdb: Path,
        ligands_sdf: Path,
        out_prefix: Path,
        device: str,
    ) -> pd.DataFrame:
        out_csv = out_prefix.with_suffix(".csv")
        if out_csv.exists():
            return pd.read_csv(out_csv)

        env, _ = visible_gpu_env(device)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.repo) if not existing_pp else f"{self.repo}:{existing_pp}"
        env.setdefault("RTMSCORE_NUM_WORKERS", "0")
        env.setdefault("DGLBACKEND", "pytorch")
        rtmscore_home = self._cfg.paths.cache_dir / "rtmscore_home"
        dgl_cache = self._cfg.paths.cache_dir / "dgl"
        rtmscore_home.mkdir(parents=True, exist_ok=True)
        dgl_cache.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(rtmscore_home)
        env["DGL_DOWNLOAD_DIR"] = str(dgl_cache)
        env["XDG_CACHE_HOME"] = str(self._cfg.paths.cache_dir)
        cmd = [
            str(self.python), str(self.script),
            "-p", str(pocket_pdb),
            "-l", str(ligands_sdf),
            "-m", str(self.model),
            "-o", str(out_prefix),
            "-c", str(self._cfg.rtmscore_cutoff_angstrom),
        ]
        if self._cfg.rtmscore_parallel_graphs:
            cmd.append("-pl")
        run_command(cmd, cwd=self.repo, env=env, stream=True, quiet=self._cfg.quiet)
        if not out_csv.exists():
            raise RuntimeError(f"RTMScore did not produce expected output: {out_csv}")
        return pd.read_csv(out_csv)

    def score(
        self,
        pose_manifest: pd.DataFrame,
        ligand_sdfs: dict[str, Path],
        pocket_specs: list[PocketSpec],
        score_dir: Path,
    ) -> pd.DataFrame:
        if pose_manifest.empty:
            return pd.DataFrame(columns=["pose_id", "rtmscore_score"])
        if not self.ready():
            missing = "\n  ".join(self.requirements())
            raise RuntimeError(
                "RTMScore is not installed/configured. Missing:\n  "
                f"{missing}\n"
                "Run: bash setup.sh rtmscore\n"
                "Expected repo at models/rtmscore, env at envs/uv/rtmscore, "
                "and trained model under models/rtmscore/trained_models/."
            )

        score_dir.mkdir(parents=True, exist_ok=True)
        spec_by_pocket = {spec.pocket_id: spec for spec in pocket_specs}
        devices = self.get_devices()
        pending = [
            pid for pid in sorted(ligand_sdfs)
            if pid in spec_by_pocket and not (score_dir / f"{pid}_scores.csv").exists()
        ]
        cached = [
            pid for pid in sorted(ligand_sdfs)
            if pid in spec_by_pocket and (score_dir / f"{pid}_scores.csv").exists()
        ]
        cached_n = int(pose_manifest[pose_manifest["pocket_id"].isin(cached)].shape[0])
        pending_n = int(pose_manifest[pose_manifest["pocket_id"].isin(pending)].shape[0])

        print(
            f"RTMScore scoring {pending_n} ligand poses across {len(devices)} GPU(s) "
            f"({len(pending)} pocket batch(es)); {cached_n} cached",
            flush=True,
        )

        progress_lock = threading.Lock()
        with tqdm(
            total=len(pose_manifest),
            initial=cached_n,
            desc="RTMScore scored poses",
            unit="pose",
            dynamic_ncols=True,
        ) as bar:

            def _run_one(task_idx: int, pocket_id: str) -> tuple[str, pd.DataFrame]:
                spec = spec_by_pocket[pocket_id]
                out_prefix = score_dir / f"{pocket_id}_scores"
                df = self._score_pocket(
                    pocket_id=pocket_id,
                    pocket_pdb=spec.pocket_pdb,
                    ligands_sdf=ligand_sdfs[pocket_id],
                    out_prefix=out_prefix,
                    device=devices[task_idx % len(devices)],
                )
                with progress_lock:
                    n_rows = int(pose_manifest[pose_manifest["pocket_id"] == pocket_id].shape[0])
                    bar.update(n_rows)
                return pocket_id, df

            if pending:
                max_workers = min(len(devices), len(pending))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_run_one, task_idx, pocket_id): pocket_id
                        for task_idx, pocket_id in enumerate(pending)
                    }
                    for future in as_completed(futures):
                        pocket_id = futures[future]
                        try:
                            future.result()
                        except Exception as exc:
                            print(f"RTMScore {pocket_id} failed: {exc}", flush=True)

        score_frames: list[pd.DataFrame] = []
        for pocket_id in sorted(ligand_sdfs):
            score_csv = score_dir / f"{pocket_id}_scores.csv"
            if not score_csv.exists():
                continue
            frame = pd.read_csv(score_csv)
            if "score" not in frame.columns:
                continue
            frame = frame.rename(columns={"score": "rtmscore_score"})
            manifest_ids = pose_manifest.loc[
                pose_manifest["pocket_id"] == pocket_id, "pose_id"
            ].astype(str).tolist()
            manifest_id_set = set(manifest_ids)
            if "id" in frame.columns:
                raw_ids = frame["id"].astype(str)
                stripped_ids = raw_ids.str.replace(r"-\d+$", "", regex=True)
                index_to_pose = {str(i): pose_id for i, pose_id in enumerate(manifest_ids)}
                if set(raw_ids).issubset(manifest_id_set):
                    frame["pose_id"] = raw_ids
                elif set(stripped_ids).issubset(manifest_id_set):
                    frame["pose_id"] = stripped_ids
                elif set(raw_ids).issubset(index_to_pose):
                    frame["pose_id"] = raw_ids.map(index_to_pose)
                else:
                    frame["pose_id"] = raw_ids
            elif len(frame) == len(manifest_ids):
                frame["pose_id"] = manifest_ids
            else:
                continue
            frame["pocket_id"] = pocket_id
            score_frames.append(frame[["pose_id", "pocket_id", "rtmscore_score"]])
        if not score_frames:
            return pd.DataFrame(columns=["pose_id", "pocket_id", "rtmscore_score"])
        return pd.concat(score_frames, ignore_index=True)


def usable_rtmscore_scores(frame: pd.DataFrame) -> bool:
    return (
        not frame.empty
        and "rtmscore_score" in frame.columns
        and pd.to_numeric(frame["rtmscore_score"], errors="coerce").notna().any()
    )


def clear_rtmscore_score_cache(score_dir: Path) -> None:
    """Remove score artifacts while preserving RTMScore input SDFs and manifests."""
    for path in score_dir.glob("pocket_*_scores.csv"):
        path.unlink(missing_ok=True)
    for name in ("rtmscore_pose_scores.csv", "rtmscore_pose_level.csv"):
        (score_dir / name).unlink(missing_ok=True)


def aggregate_rtmscore_scores(
    unique_df: pd.DataFrame,
    pose_manifest: pd.DataFrame,
    pose_scores: pd.DataFrame,
) -> pd.DataFrame:
    if pose_manifest.empty or pose_scores.empty:
        merged = unique_df.copy()
        merged["rank_score"] = 0.0
        merged["rtmscore_score"] = np.nan
        merged["rtmscore_n_poses"] = 0
        return merged

    scored_poses = pose_manifest.merge(pose_scores, on=["pose_id", "pocket_id"], how="inner")
    scored_poses["rtmscore_score"] = pd.to_numeric(scored_poses["rtmscore_score"], errors="coerce")
    scored_poses = scored_poses.dropna(subset=["rtmscore_score"])
    if scored_poses.empty:
        merged = unique_df.copy()
        merged["rank_score"] = 0.0
        merged["rtmscore_score"] = np.nan
        merged["rtmscore_n_poses"] = 0
        return merged

    best_idx = scored_poses.groupby("smiles")["rtmscore_score"].idxmax()
    best = scored_poses.loc[best_idx, [
        "smiles", "pose_id", "generator", "pocket_id", "rtmscore_score",
    ]].rename(columns={
        "pose_id": "rtmscore_best_pose_id",
        "generator": "rtmscore_best_generator",
        "pocket_id": "rtmscore_best_pocket_id",
    })
    agg = scored_poses.groupby("smiles", sort=False).agg(
        rtmscore_mean_score=("rtmscore_score", "mean"),
        rtmscore_n_poses=("rtmscore_score", "count"),
    ).reset_index()
    agg = agg.merge(best, on="smiles", how="left")

    merged = unique_df.merge(agg, on="smiles", how="left")
    merged["rank_score"] = merged["rtmscore_score"].fillna(0.0)
    merged["rtmscore_n_poses"] = merged["rtmscore_n_poses"].fillna(0).astype(int)
    return merged


def score_with_rtmscore(
    generated_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    pocket_specs: list[PocketSpec],
    run_dir: Path,
    cfg: Config,
) -> pd.DataFrame:
    """Score saved generator poses with RTMScore and merge pose scores by SMILES."""
    pose_manifest, ligand_sdfs = build_rtmscore_pose_batches(run_dir, generated_df)
    if pose_manifest.empty:
        raise RuntimeError(
            "No RTMScore-compatible pose SDFs were found. "
            "RTMScore needs saved 3D ligand poses under generator_outputs/."
        )

    score_dir = run_dir / "rtmscore"
    pose_scores_path = score_dir / "rtmscore_pose_scores.csv"
    if pose_scores_path.exists():
        print(f"\nRTMScore scoring: resuming from {pose_scores_path.name}", flush=True)
        pose_scores = pd.read_csv(pose_scores_path)
        if not usable_rtmscore_scores(pose_scores):
            print("Cached RTMScore pose scores are invalid; overwriting score cache.", flush=True)
            clear_rtmscore_score_cache(score_dir)
            pose_scores = pd.DataFrame()
    else:
        pose_scores = pd.DataFrame()

    if pose_scores.empty:
        oracle = RTMScorePoseOracle(cfg)
        pose_scores = oracle.score(
            pose_manifest=pose_manifest,
            ligand_sdfs=ligand_sdfs,
            pocket_specs=pocket_specs,
            score_dir=score_dir,
        )
        if not usable_rtmscore_scores(pose_scores):
            raise RuntimeError(
                "RTMScore produced no usable scores. The run was not marked scored; "
                "fix the RTMScore/DGL environment and rerun with the same RUN_PREFIX."
            )
        pose_scores.to_csv(pose_scores_path, index=False)

    pose_level = pose_manifest.merge(pose_scores, on=["pose_id", "pocket_id"], how="left")
    pose_level.to_csv(score_dir / "rtmscore_pose_level.csv", index=False)

    merged = aggregate_rtmscore_scores(unique_df, pose_manifest, pose_scores)
    merged["scorer"] = "rtmscore"
    return merged




def score_candidates(
    generated_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    pocket_specs: list[PocketSpec],
    run_dir: Path,
    cfg: Config,
) -> pd.DataFrame:
    """Score candidates with the configured scorer and return the merged table."""
    scorer = cfg.scorer.strip().lower()
    all_smiles = unique_df["smiles"].tolist()
    if scorer == "rtmscore":
        print(f"\nRTMScore scoring poses for {len(all_smiles)} unique molecules ...", flush=True)
        return score_with_rtmscore(
            generated_df=generated_df,
            unique_df=unique_df,
            pocket_specs=pocket_specs,
            run_dir=run_dir,
            cfg=cfg,
        )
    if scorer in ("none", "skip"):
        print("\nScoring skipped; ranking by QED only.", flush=True)
        merged = unique_df.copy()
        merged["rank_score"] = merged["qed"].fillna(0.0)
        merged["scorer"] = "none"
        return merged
    raise ValueError(f"Unknown scorer: {cfg.scorer!r}. Expected rtmscore or none.")
