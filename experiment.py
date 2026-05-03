"""Top-level experiment orchestration for Chorus."""
from __future__ import annotations

import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from chorus.config import Config
from chorus.generators import BasePocketGenerator, build_generators, get_scaffold, ligand_properties
from chorus.pockets import PocketSpec, prepare_target_pockets, write_pocket_spec_json
from chorus.runtime import discover_devices
from chorus.scoring import clear_rtmscore_score_cache, score_candidates

try:
    import torch
except ImportError:
    torch = None

@dataclass
class MolRecord:
    smiles: str
    generator: str
    pocket_id: str
    scaffold: str
    mw: float
    logp: float
    qed: float
    hbd: float
    hba: float
    rot_bonds: float
    rings: float
    source_rank: int


def build_generated_dataframe(
    results: dict[tuple[str, str], list[str]],
) -> pd.DataFrame:
    """Convert per-generator SMILES lists into a source-level molecule table."""
    rows: list[dict[str, Any]] = []
    for (generator, pocket_id), smiles_list in results.items():
        seen: set[str] = set()
        for rank, smi in enumerate(smiles_list, start=1):
            if smi in seen:
                continue
            seen.add(smi)
            props = ligand_properties(smi)
            if not props:
                continue
            rows.append(dataclasses.asdict(MolRecord(
                smiles=smi, generator=generator, pocket_id=pocket_id,
                scaffold=get_scaffold(smi),
                mw=float(props["mw"]), logp=float(props["logp"]),
                qed=float(props["qed"]), hbd=float(props["hbd"]),
                hba=float(props["hba"]), rot_bonds=float(props["rot_bonds"]),
                rings=float(props["rings"]), source_rank=rank,
            )))
    if not rows:
        return pd.DataFrame(columns=[f.name for f in MolRecord.__dataclass_fields__.values()])
    return pd.DataFrame(rows)


def build_unique_dataframe(generated_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate SMILES while preserving generator and pocket provenance."""
    if generated_df.empty:
        return generated_df.copy()
    unique_df = generated_df.groupby("smiles", sort=False).agg(
        scaffold=("scaffold", "first"),
        mw=("mw", "first"),
        logp=("logp", "first"),
        qed=("qed", "first"),
        hbd=("hbd", "first"),
        hba=("hba", "first"),
        rot_bonds=("rot_bonds", "first"),
        rings=("rings", "first"),
        generators=("generator", lambda s: ", ".join(sorted(set(s)))),
        pocket_ids=("pocket_id", lambda s: ", ".join(sorted(set(s)))),
        n_generators=("generator", "nunique"),
        n_pockets=("pocket_id", "nunique"),
        source_min_rank=("source_rank", "min"),
    ).reset_index()
    primary = (
        generated_df.sort_values(["smiles", "source_rank", "generator", "pocket_id"])
        .drop_duplicates("smiles")
        [["smiles", "generator", "pocket_id"]]
        .rename(columns={"generator": "primary_generator", "pocket_id": "primary_pocket_id"})
    )
    return unique_df.merge(primary, on="smiles", how="left")


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def get_devices() -> list[str]:
    devices = discover_devices()
    if devices != ["cpu"]:
        return devices
    if torch is not None and torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return devices if devices else ["cpu"]
    return ["cpu"]


WORKERS_PER_GPU: dict[str, int] = {
    "DiffSBDD":      1,
    "DiffSBDDJoint": 1,
    "PocketXMol":    1,
    "PocketXMolAR":  1,
}


def scored_candidates_path(run_dir: Path, scorer: str) -> Path:
    return run_dir / f"scored_candidates_{scorer}.csv"


def scored_cache_is_usable(frame: pd.DataFrame, scorer: str) -> bool:
    if frame.empty:
        return False
    if scorer == "rtmscore":
        has_scores = (
            "rtmscore_score" in frame.columns
            and pd.to_numeric(frame["rtmscore_score"], errors="coerce").notna().any()
        )
        has_pose_counts = (
            "rtmscore_n_poses" in frame.columns
            and pd.to_numeric(frame["rtmscore_n_poses"], errors="coerce").fillna(0).sum() > 0
        )
        return bool(has_scores and has_pose_counts)
    return "rank_score" in frame.columns


def clear_scored_cache(run_dir: Path, scorer: str) -> None:
    scored_candidates_path(run_dir, scorer).unlink(missing_ok=True)
    if scorer == "rtmscore":
        clear_rtmscore_score_cache(run_dir / "rtmscore")



def run_experiment(cfg: Config, run_name: str | None = None, anchor_residue: str | None = None) -> Path:
    """Run or resume one target experiment and rebuild its analysis outputs."""
    from chorus.analysis import load_cached_run_tables, rebuild_analysis_outputs

    run_name = run_name or f"{cfg.pdb_id.lower()}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = cfg.paths.results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.makedirs()
    scorer = cfg.scorer.strip().lower()

    cached = load_cached_run_tables(run_dir, scorer)
    if cached is not None:
        generated_df, unique_df, merged, pocket_specs = cached
        print("Rebuilding analysis CSVs and figures from cached run outputs.", flush=True)
        return rebuild_analysis_outputs(
            run_dir=run_dir,
            run_name=run_name,
            cfg=cfg,
            generated_df=generated_df,
            scored_df=merged,
            pocket_specs=pocket_specs,
            generator_errors={},
            scorer=scorer,
        )

    # Stage 1: detect pockets once, then cache the exact pocket specs for rebuilds.
    pocket_specs = prepare_target_pockets(cfg.pdb_id, cfg, anchor_residue=anchor_residue)
    pocket_dir = run_dir / "pocket_specs"
    pocket_dir.mkdir(exist_ok=True)
    for spec in pocket_specs:
        write_pocket_spec_json(spec, pocket_dir / f"{spec.pocket_id}.json")

    # Stage 2: run one generator across all pockets at a time so per-model CSVs
    # are usable resume points if a later generator or scorer fails.
    all_results: dict[tuple[str, str], list[str]] = {}
    generator_errors: dict[str, str] = {}
    generator_output_root = run_dir / "generator_outputs"
    generator_output_root.mkdir(exist_ok=True)

    first_generators = build_generators(pocket_specs[0], cfg)
    generator_names = list(first_generators.keys())
    devices = get_devices()
    print(f"Detected devices: {', '.join(devices)}", flush=True)

    for gen_name in generator_names:
        intermediate_csv = run_dir / f"generated_{gen_name.lower()}.csv"

        if intermediate_csv.exists():
            print(f"\n=== {gen_name}: resuming from {intermediate_csv.name} ===", flush=True)
            saved = pd.read_csv(intermediate_csv)
            for pid in saved["pocket_id"].unique():
                smiles = saved.loc[saved["pocket_id"] == pid, "smiles"].tolist()
                all_results[(gen_name, pid)] = smiles
                print(f"  {gen_name} @ {pid}: loaded {len(smiles)} molecules from CSV", flush=True)
            continue

        n_pockets = len(pocket_specs)
        total_tasks = n_pockets
        workers_per_gpu = WORKERS_PER_GPU.get(gen_name, 1)
        max_workers = min(total_tasks, len(devices) * workers_per_gpu)
        active_device_count = min(
            len(devices),
            max(1, (max_workers + workers_per_gpu - 1) // workers_per_gpu),
        )
        active_devices = devices[:active_device_count]

        print(
            f"\n=== {gen_name}: {n_pockets} pockets × {cfg.n_generate_per_model_per_pocket} samples "
            f"({len(active_devices)} device(s) × {workers_per_gpu} worker(s)/device = {max_workers} parallel; "
            f"using {', '.join(active_devices)}) ===",
            flush=True,
        )

        pocket_generators: list[tuple[PocketSpec, BasePocketGenerator]] = []
        for i, spec in enumerate(pocket_specs):
            generators = build_generators(spec, cfg)
            gen = generators[gen_name]
            gen.device = active_devices[i % len(active_devices)]
            pocket_generators.append((spec, gen))

        def _run_pocket(
            name: str, gen: BasePocketGenerator, pocket_id: str, n: int,
        ) -> tuple[str, str, list[str], float]:
            t0 = time.time()
            out = generator_output_root / name.lower() / pocket_id
            out.mkdir(parents=True, exist_ok=True)
            smiles = gen.generate_validated(n, out_dir=out)
            return name, pocket_id, smiles, time.time() - t0

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_pocket, gen_name, gen, spec.pocket_id,
                    cfg.n_generate_per_model_per_pocket,
                ): spec.pocket_id
                for spec, gen in pocket_generators
            }
            valid_generated = 0
            with tqdm(
                total=n_pockets * cfg.n_generate_per_model_per_pocket,
                desc=f"{gen_name} generated samples",
                unit="mol",
                dynamic_ncols=True,
            ) as generation_bar:
                for future in as_completed(futures):
                    pocket_id = futures[future]
                    completed += 1
                    try:
                        gn, pid, smiles, elapsed = future.result()
                        all_results[(gn, pid)] = smiles
                        valid_generated += len(smiles)
                        generation_bar.update(cfg.n_generate_per_model_per_pocket)
                        generation_bar.set_postfix(valid=valid_generated)
                        tqdm.write(
                            f"  {gn} @ {pid}: {len(smiles)} valid mols "
                            f"({elapsed / 60:.1f} min) [{completed}/{total_tasks}]"
                        )
                        if not smiles:
                            generator_errors[f"{gn}:{pid}"] = "Zero valid molecules"
                    except Exception as exc:
                        all_results[(gen_name, pocket_id)] = []
                        generator_errors[f"{gen_name}:{pocket_id}"] = str(exc)
                        generation_bar.update(cfg.n_generate_per_model_per_pocket)
                        generation_bar.set_postfix(valid=valid_generated)
                        tqdm.write(
                            f"  !!! {gen_name} @ {pocket_id}: {exc} [{completed}/{total_tasks}]"
                        )

        partial_df = build_generated_dataframe(
            {k: v for k, v in all_results.items() if k[0] == gen_name}
        )
        if not partial_df.empty:
            partial_df.to_csv(intermediate_csv, index=False)
            print(f"  Saved {gen_name} results to {intermediate_csv.name}", flush=True)

    # Stage 3: normalize generator outputs into source-level and unique tables.
    generated_df = build_generated_dataframe(all_results)
    if generated_df.empty:
        print("WARNING: No molecules generated by any model.")
        (run_dir / "run_summary.json").write_text(
            json.dumps({"error": "No molecules", "generator_errors": generator_errors}, indent=2)
        )
        return run_dir
    generated_df.to_csv(run_dir / "generated_by_generator.csv", index=False)

    unique_df = build_unique_dataframe(generated_df)
    unique_df.to_csv(run_dir / "unique_generated.csv", index=False)

    # Stage 4: score generated poses or use a lightweight fallback score.
    scored_path = scored_candidates_path(run_dir, scorer)
    if scored_path.exists():
        print(f"\n{scorer} scoring: resuming from {scored_path.name}", flush=True)
        merged = pd.read_csv(scored_path)
        if not scored_cache_is_usable(merged, scorer):
            print(f"Cached {scored_path.name} is not usable; overwriting scoring cache.", flush=True)
            clear_scored_cache(run_dir, scorer)
            merged = score_candidates(generated_df, unique_df, pocket_specs, run_dir, cfg)
            merged.to_csv(scored_path, index=False)
    else:
        merged = score_candidates(generated_df, unique_df, pocket_specs, run_dir, cfg)
        merged.to_csv(scored_path, index=False)

    out = rebuild_analysis_outputs(
        run_dir=run_dir,
        run_name=run_name,
        cfg=cfg,
        generated_df=generated_df,
        scored_df=merged,
        pocket_specs=pocket_specs,
        generator_errors=generator_errors,
        scorer=scorer,
    )

    print(f"\nRun complete: {run_dir}")
    print(f"Pockets: {len(pocket_specs)}")
    print(f"Generated rows: {len(generated_df):,}")
    print(f"Unique molecules: {merged['smiles'].nunique():,}")
    top_hits = pd.read_csv(run_dir / "top_unique_hits.csv") if (run_dir / "top_unique_hits.csv").exists() else pd.DataFrame()
    if not top_hits.empty:
        first = top_hits.iloc[0]
        score = first.get("rank_score") or 0.0
        print(f"Top hit: {first['smiles']} | generators={first['generators']} | score={score:.3f}")
    return out
