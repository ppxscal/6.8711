"""
pipeline.py — config, pocket detection, Boltz oracle, and the main
run_pipeline() orchestration function.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import shutil
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from chorus.generators import (
    BasePocketGenerator, build_generators,
    get_scaffold, hash_smiles, ligand_properties,
    fp_array, mol_from_smiles, morgan_fp,
    env_binary, env_python, run_command, visible_gpu_env,
)

try:
    import torch
except ImportError:
    torch = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None  # type: ignore[assignment]

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = Path("/data2/ppxscal")


@dataclass(frozen=True)
class Paths:
    root:            Path
    data_dir:        Path
    results_dir:     Path
    models_dir:      Path
    checkpoints_dir: Path
    tools_dir:       Path
    envs_dir:        Path
    cache_dir:       Path
    boltz_cache_dir: Path
    msa_cache_dir:   Path
    p2rank_dir:      Path

    @classmethod
    def from_root(cls, root: Path | str) -> "Paths":
        root = Path(root).resolve()
        cache = root / "cache"
        boltz_cache = cache / "boltz"
        return cls(
            root=root,
            data_dir=root / "data",
            results_dir=root / "results",
            models_dir=root / "models",
            checkpoints_dir=root / "checkpoints",
            tools_dir=root / "tools",
            envs_dir=root / "envs",
            cache_dir=cache,
            boltz_cache_dir=boltz_cache,
            msa_cache_dir=boltz_cache / "msa",
            p2rank_dir=root / "tools" / "p2rank",
        )

    def makedirs(self) -> None:
        for path in (
            self.data_dir, self.results_dir, self.models_dir,
            self.checkpoints_dir, self.tools_dir, self.envs_dir,
            self.cache_dir, self.boltz_cache_dir, self.msa_cache_dir,
            self.p2rank_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    seed:            int = 42
    target_name:     str = "unknown"
    pdb_id:          str = ""

    generator_mode:      str  = "real"
    allow_mock_fallback: bool = False
    generators: tuple[str, ...] = ("DiffSBDD", "PocketXMol")

    n_generate_per_model_per_pocket: int = 1250  # 2 gens × 4 pockets × 1250 = 10k total

    pocket_radius_angstrom:    float      = 10.0
    pocket_bbox_size_angstrom: float      = 23.0
    p2rank_min_score:          float      = 0.5
    p2rank_max_pockets:        int        = 4
    ligand_resname_preference: str | None = None

    boltz_msa_server_url:       str = "https://api.colabfold.com"
    boltz_msa_pairing_strategy: str = "greedy"
    boltz_timeout_sec:          int = 600

    scorer: str = "rtmscore"
    rtmscore_model_name: str = "rtmscore_model1.pth"
    rtmscore_cutoff_angstrom: float = 10.0
    rtmscore_parallel_graphs: bool = False

    max_pca_points: int = 5000
    max_umap_points: int = 1000
    max_cluster_points: int = 1000
    ecfp_family_sim_threshold: float = 0.30
    max_tanimoto_refs_per_pocket: int = 500
    tanimoto_top_k: int = 10
    quiet:          bool = True

    paths: Paths = field(default_factory=lambda: Paths.from_root(_DEFAULT_ROOT))


# ---------------------------------------------------------------------------
# Pocket detection
# ---------------------------------------------------------------------------

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}

COMMON_NON_LIGANDS = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "PEG", "EDO", "GOL", "ACT", "FMT", "EOH", "DMS",
    "MES", "TRS", "NAG", "MAN", "BMA", "CL", "NA", "K", "CA", "MG", "ZN",
}

KNOWN_POCKET_HINTS: dict[str, list[str]] = {
    "2PN7": ["A:98", "B:98"],
}

P2RANK_VERSION = "2.4.2"
P2RANK_URL = (
    f"https://github.com/rdk/p2rank/releases/download/{P2RANK_VERSION}"
    f"/p2rank_{P2RANK_VERSION}.tar.gz"
)


@dataclass
class LigandCandidate:
    chain: str
    resname: str
    resseq: str
    atom_lines: list[str]
    coords: np.ndarray

    @property
    def residue_id(self) -> str:
        return f"{self.chain}:{self.resseq}"

    @property
    def centroid(self) -> tuple[float, float, float]:
        return tuple(np.mean(self.coords, axis=0).tolist())  # type: ignore[return-value]


@dataclass
class PocketSpec:
    pocket_id: str
    full_pdb: Path
    pocket_pdb: Path
    ligand_residue_id: str
    ligand_resname: str
    protein_chain: str
    protein_sequence: str
    center: tuple[float, float, float]
    bbox_size: float
    contact_residues: list[str]
    pocket_source: str
    has_reference_ligand: bool
    p2rank_score: float = 0.0
    p2rank_residues: list[str] = field(default_factory=list)


def pdb_atom_coord(line: str) -> np.ndarray:
    try:
        return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
    except ValueError:
        pass
    try:
        return np.array([float(line[31:39]), float(line[39:47]), float(line[47:55])], dtype=float)
    except ValueError:
        pass
    parts = line.split()
    return np.array([float(parts[6]), float(parts[7]), float(parts[8])], dtype=float)


def download_pdb(pdb_id: str, cfg: Config) -> Path:
    out_path = cfg.paths.data_dir / f"{pdb_id.lower()}.pdb"
    if out_path.exists():
        return out_path
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id} from {pdb_url}")
    tmp_path = out_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(pdb_url, tmp_path)
        first_line = tmp_path.read_text(errors="ignore")[:20]
        if first_line.startswith("<!"):
            raise ValueError("Got HTML instead of PDB")
        tmp_path.rename(out_path)
        return out_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
    cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    cif_path = out_path.with_suffix(".cif")
    print(f"  .pdb not available, downloading mmCIF from {cif_url}")
    urllib.request.urlretrieve(cif_url, cif_path)
    try:
        import gemmi
        st = gemmi.read_structure(str(cif_path))
        st.write_pdb(str(out_path))
        return out_path
    except ImportError:
        pass
    from Bio.PDB import MMCIFParser, PDBIO
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(cif_path))
    io_out = PDBIO()
    io_out.set_structure(structure)
    io_out.save(str(out_path))
    return out_path


def residue_atom_groups(pdb_path: Path) -> dict[tuple[str, str, str], list[str]]:
    groups: dict[tuple[str, str, str], list[str]] = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip() or "A"
            resname = line[17:20].strip()
            resseq = line[22:26].strip()
            groups.setdefault((chain, resname, resseq), []).append(line.rstrip("\n"))
    return groups


def find_ligand_candidates(pdb_path: Path) -> list[LigandCandidate]:
    groups: dict[tuple[str, str, str], list[str]] = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            chain = line[21].strip() or "A"
            resseq = line[22:26].strip()
            element = line[76:78].strip().upper()
            if resname in COMMON_NON_LIGANDS or element == "H":
                continue
            groups.setdefault((chain, resname, resseq), []).append(line.rstrip("\n"))
    ligands = []
    for (chain, resname, resseq), atom_lines in groups.items():
        coords = np.array([pdb_atom_coord(line) for line in atom_lines], dtype=float)
        if len(coords) < 5:
            continue
        ligands.append(LigandCandidate(
            chain=chain, resname=resname, resseq=resseq,
            atom_lines=atom_lines, coords=coords,
        ))
    ligands.sort(key=lambda lig: (-len(lig.atom_lines), lig.resname, lig.residue_id))
    return ligands


def select_primary_ligand(
    candidates: list[LigandCandidate], preferred_resname: str | None = None,
) -> LigandCandidate:
    if not candidates:
        raise RuntimeError("No suitable bound ligand found in the PDB.")
    if preferred_resname:
        for c in candidates:
            if c.resname == preferred_resname:
                return c
    return candidates[0]


def extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    seen: list[str] = []
    seen_residues: set[tuple[str, str, str]] = set()
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip() or "A"
            if chain != chain_id:
                continue
            resname = line[17:20].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            if resname in AA3_TO_1:
                seen.append(AA3_TO_1[resname])
    if not seen:
        raise RuntimeError(f"Could not extract sequence for chain {chain_id}")
    return "".join(seen)


def residue_distance_to_ligand(atom_lines: list[str], ligand_coords: np.ndarray) -> float:
    coords = np.array([pdb_atom_coord(line) for line in atom_lines], dtype=float)
    dists = np.linalg.norm(coords[:, None, :] - ligand_coords[None, :, :], axis=-1)
    return float(dists.min())


def contact_residues_from_ligand(
    pdb_path: Path, ligand: LigandCandidate, cutoff: float = 6.0,
) -> list[str]:
    residues: dict[tuple[str, str], list[str]] = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip() or "A"
            resseq = line[22:26].strip()
            residues.setdefault((chain, resseq), []).append(line.rstrip("\n"))
    contacts = []
    for (chain, resseq), lines in residues.items():
        if residue_distance_to_ligand(lines, ligand.coords) <= cutoff:
            contacts.append(f"{chain}:{resseq}")
    return sorted(set(contacts))


def write_pocket_pdb(
    full_pdb: Path, ligand: LigandCandidate, out_path: Path, radius: float = 10.0,
) -> Path:
    kept_lines = []
    with open(full_pdb) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip().upper()
                if element == "H":
                    continue
                if line.startswith("ATOM"):
                    coord = pdb_atom_coord(line)
                    min_dist = np.linalg.norm(ligand.coords - coord[None, :], axis=1).min()
                    if min_dist <= radius:
                        kept_lines.append(line)
            elif line.startswith(("TER", "END")):
                kept_lines.append(line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(kept_lines))
    return out_path


def write_pocket_spec_json(spec: PocketSpec, out_path: Path) -> None:
    payload = {
        "pocket_id": spec.pocket_id,
        "full_pdb": str(spec.full_pdb),
        "pocket_pdb": str(spec.pocket_pdb),
        "ligand_residue_id": spec.ligand_residue_id,
        "ligand_resname": spec.ligand_resname,
        "protein_chain": spec.protein_chain,
        "sequence_length": len(spec.protein_sequence),
        "center": list(spec.center),
        "bbox_size": spec.bbox_size,
        "contact_residues": spec.contact_residues,
        "pocket_source": spec.pocket_source,
        "has_reference_ligand": spec.has_reference_ligand,
        "p2rank_score": spec.p2rank_score,
        "p2rank_residues": spec.p2rank_residues,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def read_pocket_spec_json(path: Path) -> PocketSpec:
    payload = json.loads(path.read_text())
    return PocketSpec(
        pocket_id=str(payload.get("pocket_id", path.stem)),
        full_pdb=Path(payload.get("full_pdb", "")),
        pocket_pdb=Path(payload.get("pocket_pdb", "")),
        ligand_residue_id=str(payload.get("ligand_residue_id", "")),
        ligand_resname=str(payload.get("ligand_resname", "")),
        protein_chain=str(payload.get("protein_chain", "")),
        protein_sequence=str(payload.get("protein_sequence", "")),
        center=tuple(float(x) for x in payload.get("center", [0.0, 0.0, 0.0])),  # type: ignore[arg-type]
        bbox_size=float(payload.get("bbox_size", 0.0)),
        contact_residues=[str(x) for x in payload.get("contact_residues", [])],
        pocket_source=str(payload.get("pocket_source", "cached")),
        has_reference_ligand=bool(payload.get("has_reference_ligand", False)),
        p2rank_score=float(payload.get("p2rank_score", 0.0)),
        p2rank_residues=[str(x) for x in payload.get("p2rank_residues", [])],
    )


def read_cached_pocket_specs(run_dir: Path) -> list[PocketSpec]:
    pocket_dir = run_dir / "pocket_specs"
    if not pocket_dir.exists():
        return []
    specs = []
    for path in sorted(pocket_dir.glob("*.json")):
        try:
            specs.append(read_pocket_spec_json(path))
        except Exception as exc:
            print(f"WARNING: could not read cached pocket spec {path.name}: {exc}", flush=True)
    return specs


def ensure_p2rank(cfg: Config) -> Path:
    prank_bin = cfg.paths.p2rank_dir / f"p2rank_{P2RANK_VERSION}" / "prank"
    if prank_bin.exists():
        return prank_bin
    cfg.paths.p2rank_dir.mkdir(parents=True, exist_ok=True)
    archive = cfg.paths.p2rank_dir / f"p2rank_{P2RANK_VERSION}.tar.gz"
    if not archive.exists():
        print(f"Downloading P2Rank {P2RANK_VERSION}")
        urllib.request.urlretrieve(P2RANK_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(cfg.paths.p2rank_dir)
    if not prank_bin.exists():
        raise RuntimeError(f"P2Rank extraction failed: expected {prank_bin}")
    return prank_bin


def run_p2rank(pdb_path: Path, cfg: Config) -> list[dict]:
    prank_bin = ensure_p2rank(cfg)
    out_dir = pdb_path.parent / f"{pdb_path.stem}_p2rank"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_command([str(prank_bin), "predict", "-f", str(pdb_path), "-o", str(out_dir)],
                quiet=cfg.quiet)
    predictions_csv: Path | None = None
    for candidate in out_dir.rglob("*_predictions.csv"):
        predictions_csv = candidate
        break
    if predictions_csv is None:
        raise RuntimeError(f"P2Rank produced no predictions CSV in {out_dir}")
    pockets = []
    with open(predictions_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            pockets.append({
                "name": clean.get("name", ""),
                "rank": int(clean.get("rank", 0)),
                "score": float(clean.get("score", 0.0)),
                "probability": float(clean.get("probability", 0.0)),
                "center_x": float(clean.get("center_x", 0.0)),
                "center_y": float(clean.get("center_y", 0.0)),
                "center_z": float(clean.get("center_z", 0.0)),
                "residue_ids": clean.get("residue_ids", ""),
            })
    return pockets


def _parse_p2rank_residues(residue_str: str) -> list[str]:
    parts = []
    for token in residue_str.split():
        if "_" in token:
            chain, resseq = token.rsplit("_", 1)
            parts.append(f"{chain}:{resseq}")
    return parts


def _ligand_from_pocket_residues(
    pdb_path: Path,
    center: tuple[float, float, float],
    pocket_residues: list[str],
    pocket_name: str,
) -> LigandCandidate:
    groups = residue_atom_groups(pdb_path)
    pocket_res_set = set(pocket_residues)
    atom_coords: list[np.ndarray] = []
    for (chain, resname, resseq), atom_lines in groups.items():
        if f"{chain}:{resseq}" in pocket_res_set:
            for line in atom_lines:
                atom_coords.append(pdb_atom_coord(line))
    coords = (
        np.array(atom_coords, dtype=float)
        if atom_coords
        else np.array([list(center)], dtype=float)
    )
    chain = pocket_residues[0].split(":")[0] if pocket_residues else "A"
    return LigandCandidate(chain=chain, resname=pocket_name, resseq="0",
                           atom_lines=[], coords=coords)


def detect_pockets(pdb_path: Path, cfg: Config) -> list[PocketSpec]:
    raw = run_p2rank(pdb_path, cfg)
    if not raw:
        raise RuntimeError(f"P2Rank found no pockets in {pdb_path}")
    filtered = [p for p in raw if p["score"] >= cfg.p2rank_min_score] or [raw[0]]
    filtered = sorted(filtered, key=lambda p: -p["score"])[:cfg.p2rank_max_pockets]
    first_residues = _parse_p2rank_residues(filtered[0]["residue_ids"])
    primary_chain = first_residues[0].split(":")[0] if first_residues else "A"
    seq_cache: dict[str, str] = {}
    specs = []
    for i, pocket in enumerate(filtered):
        pocket_id = f"pocket_{i + 1}"
        center: tuple[float, float, float] = (pocket["center_x"], pocket["center_y"], pocket["center_z"])
        p2rank_residues = _parse_p2rank_residues(pocket["residue_ids"])
        pocket_chain = p2rank_residues[0].split(":")[0] if p2rank_residues else primary_chain
        if pocket_chain not in seq_cache:
            try:
                seq_cache[pocket_chain] = extract_chain_sequence(pdb_path, pocket_chain)
            except RuntimeError:
                seq_cache[pocket_chain] = seq_cache.get(primary_chain, "")
        ligand = _ligand_from_pocket_residues(pdb_path, center, p2rank_residues, pocket["name"])
        pocket_pdb_path = (
            cfg.paths.data_dir
            / f"{pdb_path.stem}_{pocket_id}_pocket{int(cfg.pocket_radius_angstrom)}.pdb"
        )
        write_pocket_pdb(pdb_path, ligand, pocket_pdb_path, radius=cfg.pocket_radius_angstrom)
        specs.append(PocketSpec(
            pocket_id=pocket_id,
            full_pdb=pdb_path,
            pocket_pdb=pocket_pdb_path,
            ligand_residue_id=f"{pocket_chain}:p2rank_{i + 1}",
            ligand_resname=pocket["name"],
            protein_chain=pocket_chain,
            protein_sequence=seq_cache[pocket_chain],
            center=center,
            bbox_size=float(cfg.pocket_bbox_size_angstrom),
            contact_residues=p2rank_residues,
            pocket_source="p2rank",
            has_reference_ligand=False,
            p2rank_score=pocket["score"],
            p2rank_residues=p2rank_residues,
        ))
    return specs


def prepare_target_pockets(
    pdb_id: str, cfg: Config, anchor_residue: str | None = None,
) -> list[PocketSpec]:
    full_pdb = download_pdb(pdb_id, cfg)
    if anchor_residue is not None:
        return _prepare_single_pocket(pdb_id, full_pdb, cfg, anchor_residue)
    candidates = find_ligand_candidates(full_pdb)
    if candidates and cfg.ligand_resname_preference:
        return _prepare_single_pocket(pdb_id, full_pdb, cfg, anchor_residue)
    try:
        specs = detect_pockets(full_pdb, cfg)
        print(f"P2Rank detected {len(specs)} druggable pocket(s)")
        for spec in specs:
            print(
                f"  {spec.pocket_id}: score={spec.p2rank_score:.2f}, "
                f"center=({spec.center[0]:.1f}, {spec.center[1]:.1f}, {spec.center[2]:.1f}), "
                f"residues={len(spec.contact_residues)}"
            )
        return specs
    except Exception as exc:
        print(f"P2Rank failed ({exc}), falling back to single-pocket detection")
        return _prepare_single_pocket(pdb_id, full_pdb, cfg, anchor_residue)


def _prepare_single_pocket(
    pdb_id: str, full_pdb: Path, cfg: Config, anchor_residue: str | None = None,
) -> list[PocketSpec]:
    candidates = find_ligand_candidates(full_pdb)
    pocket_source = "bound_ligand"
    has_reference_ligand = True
    if candidates:
        ligand = select_primary_ligand(candidates, cfg.ligand_resname_preference)
    else:
        # Use anchor residue or known hints
        candidates_list: list[str] = []
        if anchor_residue:
            candidates_list.append(anchor_residue)
        candidates_list.extend(KNOWN_POCKET_HINTS.get(pdb_id.upper(), []))
        ligand = None
        for rid in candidates_list:
            chain, resseq = rid.split(":", 1)
            groups = residue_atom_groups(full_pdb)
            for (c, rn, rs), atom_lines in groups.items():
                if c == chain and rs == resseq:
                    coords = np.array([pdb_atom_coord(l) for l in atom_lines], dtype=float)
                    ligand = LigandCandidate(chain=c, resname=rn, resseq=rs,
                                             atom_lines=atom_lines, coords=coords)
                    break
            if ligand:
                break
        if ligand is None:
            raise RuntimeError(
                "No bound ligand and no valid anchor residue. "
                "Pass --anchor-residue CHAIN:RESSEQ."
            )
        pocket_source = "anchor_residue"
        has_reference_ligand = False
    pocket_pdb = (
        cfg.paths.data_dir
        / f"{pdb_id.lower()}_{ligand.residue_id.replace(':', '_')}_pocket{int(cfg.pocket_radius_angstrom)}.pdb"
    )
    write_pocket_pdb(full_pdb, ligand, pocket_pdb, radius=cfg.pocket_radius_angstrom)
    protein_sequence = extract_chain_sequence(full_pdb, ligand.chain)
    contact_residues = contact_residues_from_ligand(full_pdb, ligand, cutoff=6.0)
    return [PocketSpec(
        pocket_id="pocket_1",
        full_pdb=full_pdb,
        pocket_pdb=pocket_pdb,
        ligand_residue_id=ligand.residue_id,
        ligand_resname=ligand.resname,
        protein_chain=ligand.chain,
        protein_sequence=protein_sequence,
        center=ligand.centroid,
        bbox_size=float(cfg.pocket_bbox_size_angstrom),
        contact_residues=contact_residues,
        pocket_source=pocket_source,
        has_reference_ligand=has_reference_ligand,
    )]


# ---------------------------------------------------------------------------
# Boltz oracle
# ---------------------------------------------------------------------------

class BoltzCliOracle:
    """Boltz-2 affinity scorer. Distributes across all GPUs × boltz_workers_per_gpu."""

    def __init__(self, spec: PocketSpec, cfg: Config) -> None:
        self.spec = spec
        self._cfg = cfg
        boltz_bin = env_binary("boltz", "boltz", cfg.paths)
        self.binary: Path | None = boltz_bin if boltz_bin.exists() else None
        if self.binary is None:
            found = shutil.which("boltz")
            if found:
                self.binary = Path(found)
        self.python = env_python("boltz", cfg.paths)
        seq_hash = hashlib.sha256(self.spec.protein_sequence.encode()).hexdigest()[:12]
        target_slug = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in cfg.target_name.lower()
        ).strip("_") or "target"
        self.target_cache_slug = f"{target_slug}_{self.spec.protein_chain}_{seq_hash}"
        score_cache_dir = cfg.paths.boltz_cache_dir / "scores"
        score_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = score_cache_dir / f"{self.target_cache_slug}.json"
        legacy_cache_path = cfg.paths.results_dir / "boltz_cache.json"
        cache_source = self.cache_path if self.cache_path.exists() else legacy_cache_path
        self.cache: dict[str, dict[str, float | None]] = (
            json.loads(cache_source.read_text()) if cache_source.exists() else {}
        )
        self.scoring_run_dir = (
            cfg.paths.boltz_cache_dir
            / "scoring_runs"
            / f"{self.target_cache_slug}_n{cfg.n_generate_per_model_per_pocket}_affinity_s100_d1"
        )
        self.msa_cache_path = (
            cfg.paths.msa_cache_dir
            / f"{self.target_cache_slug}.csv"
        )

    def ready(self) -> bool:
        return self.binary is not None and self.python.exists()

    def _ensure_cached_msa(self) -> Path:
        if self.msa_cache_path.exists():
            return self.msa_cache_path
        if not self.python.exists():
            raise RuntimeError("Boltz Python env missing; cannot generate MSA cache.")
        sequence = self.spec.protein_sequence
        server_url = self._cfg.boltz_msa_server_url
        pairing = self._cfg.boltz_msa_pairing_strategy
        name = self.msa_cache_path.stem
        msa_dir = str(self._cfg.paths.msa_cache_dir)
        script = "\n".join([
            "from pathlib import Path",
            "from boltz.main import compute_msa",
            f"msa_dir = Path({msa_dir!r})",
            "msa_dir.mkdir(parents=True, exist_ok=True)",
            f"name = {name!r}",
            f"sequence = {sequence!r}",
            "compute_msa(",
            "    data={name: sequence},",
            "    target_id=name,",
            "    msa_dir=msa_dir,",
            f"    msa_server_url={server_url!r},",
            f"    msa_pairing_strategy={pairing!r},",
            "    msa_server_username=None,",
            "    msa_server_password=None,",
            "    api_key_header=None,",
            "    api_key_value=None,",
            ")",
            "print(msa_dir / f'{name}.csv')",
        ])
        env = os.environ.copy()
        env["BOLTZ_CACHE"] = str(self._cfg.paths.boltz_cache_dir)
        numba_cache_dir = self._cfg.paths.boltz_cache_dir / "numba"
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
        run_command([str(self.python), "-c", script], env=env, stream=True)
        if not self.msa_cache_path.exists():
            raise RuntimeError(f"Expected cached MSA at {self.msa_cache_path}")
        return self.msa_cache_path

    def _ensure_boltz_runtime_cache(self) -> None:
        """Initialize Boltz model/CCD cache before parallel GPU workers use it."""
        if not self.python.exists():
            raise RuntimeError("Boltz Python env missing; cannot initialize Boltz cache.")
        cache_dir = self._cfg.paths.boltz_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        script = "\n".join([
            "from pathlib import Path",
            "import shutil",
            "import tarfile",
            "import time",
            "from boltz.data import const",
            "from boltz.main import download_boltz2",
            f"cache = Path({str(cache_dir)!r})",
            "cache.mkdir(parents=True, exist_ok=True)",
            "mols = cache / 'mols'",
            "tar_mols = cache / 'mols.tar'",
            "def quarantine(path, reason):",
            "    if not path.exists():",
            "        return",
            "    dest = path.with_name(f'{path.name}.corrupt.{int(time.time())}')",
            "    print(f'Quarantining corrupt Boltz cache {path}: {reason}')",
            "    shutil.move(str(path), str(dest))",
            "def missing_canonicals():",
            "    return [x for x in const.canonical_tokens if not (mols / f'{x}.pkl').exists()]",
            "if mols.exists():",
            "    missing = missing_canonicals()",
            "    if missing:",
            "        quarantine(mols, 'missing canonical CCD components: ' + ', '.join(missing[:5]))",
            "        quarantine(tar_mols, 'paired with incomplete mols directory')",
            "elif tar_mols.exists():",
            "    try:",
            "        with tarfile.open(str(tar_mols), 'r') as tar:",
            "            names = {m.name for m in tar.getmembers()}",
            "        if 'mols/ALA.pkl' not in names:",
            "            quarantine(tar_mols, 'missing mols/ALA.pkl')",
            "    except Exception as exc:",
            "        quarantine(tar_mols, f'unreadable tar: {exc}')",
            "download_boltz2(cache)",
            "missing = missing_canonicals()",
            "if missing:",
            "    raise RuntimeError('Boltz CCD cache is incomplete; missing: ' + ', '.join(missing[:10]))",
            "print(f'Boltz cache ready: {cache}')",
        ])
        env = os.environ.copy()
        env["BOLTZ_CACHE"] = str(cache_dir)
        numba_cache_dir = cache_dir / "numba"
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
        run_command([str(self.python), "-c", script], env=env, stream=True, quiet=self._cfg.quiet)

    def _write_yaml(self, smiles: str, out_path: Path) -> None:
        msa_path = self._ensure_cached_msa()
        lines = [
            "version: 1", "sequences:",
            "  - protein:", "      id: A",
            f"      sequence: \"{self.spec.protein_sequence}\"",
            f"      msa: \"{msa_path}\"",
            "  - ligand:", "      id: B",
            f"      smiles: \"{smiles}\"",
            "properties:", "  - affinity:", "      binder: B", "",
        ]
        out_path.write_text("\n".join(lines))

    def _stem_for_smiles(self, smiles: str) -> str:
        return f"mol_{hashlib.sha256(smiles.encode()).hexdigest()[:16]}"

    def _write_score_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(self.cache, indent=2))
        tmp_path.replace(self.cache_path)

    def _parse_result(self, predictions_dir: Path, stem: str) -> dict[str, float | None]:
        """Parse affinity result for a single YAML stem from a batch predictions dir."""
        candidate_dir = predictions_dir / stem
        for candidate in candidate_dir.rglob("affinity_*.json") if candidate_dir.exists() else []:
            data = json.loads(candidate.read_text())
            affinity = data.get("affinity_pred_value")
            prob = data.get("affinity_probability_binary")
            rank = (
                float(prob) if prob is not None
                else (float(-affinity) if affinity is not None else 0.0)
            )
            return {
                "rank_score": rank,
                "affinity_probability_binary": float(prob) if prob is not None else None,
                "affinity_pred_value": float(affinity) if affinity is not None else None,
            }
        return {"rank_score": 0.0, "affinity_probability_binary": None, "affinity_pred_value": None}

    def _has_result_file(self, predictions_dir: Path, stem: str) -> bool:
        candidate_dir = predictions_dir / stem
        if not candidate_dir.exists():
            return False
        return any(candidate_dir.rglob("affinity_*.json"))

    def _has_structure_file(self, predictions_dir: Path, stem: str) -> bool:
        candidate_dir = predictions_dir / stem
        if not candidate_dir.exists():
            return False
        return any(candidate_dir.rglob("confidence_*.json"))

    def _parse_persistent_prediction(self, smiles: str) -> dict[str, float | None] | None:
        stem = self._stem_for_smiles(smiles)
        for predictions_dir in self.scoring_run_dir.glob("gpu_*/outputs/boltz_results_inputs/predictions"):
            if self._has_result_file(predictions_dir, stem):
                return self._parse_result(predictions_dir, stem)
        return None

    def _score_chunk(
        self,
        chunk: list[tuple[int, str]],   # (original_index, smiles)
        device: str,
        work_dir: Path,
        score_progress_callback: Callable[[int], None] | None = None,
        structure_progress_callback: Callable[[int], None] | None = None,
    ) -> list[tuple[int, str, dict[str, float | None]]]:
        """Score a chunk of molecules on one GPU by calling boltz predict once."""
        if self.binary is None:
            raise RuntimeError("Boltz binary not found.")

        yaml_dir = work_dir / "inputs"
        yaml_dir.mkdir(parents=True, exist_ok=True)
        out_dir = work_dir / "outputs"
        out_dir.mkdir(exist_ok=True)

        # Map filename stem → (original_index, smiles)
        stem_map: dict[str, tuple[int, str]] = {}
        for idx, smi in chunk:
            stem = self._stem_for_smiles(smi)
            self._write_yaml(smi, yaml_dir / f"{stem}.yaml")
            stem_map[stem] = (idx, smi)

        env, _ = visible_gpu_env(device)
        numba_cache_dir = self._cfg.paths.boltz_cache_dir / "numba"
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
        # boltz predict <dir> writes to out_dir/boltz_results_<dir.stem>/predictions/
        predictions_dir = out_dir / f"boltz_results_{yaml_dir.name}" / "predictions"

        reported_scores: set[str] = set()
        reported_structures: set[str] = set()
        stop_monitor = threading.Event()

        def _monitor_results() -> None:
            while not stop_monitor.is_set():
                newly_structured = [
                    stem for stem in stem_map
                    if stem not in reported_structures and self._has_structure_file(predictions_dir, stem)
                ]
                if newly_structured:
                    reported_structures.update(newly_structured)
                    if structure_progress_callback is not None:
                        structure_progress_callback(len(newly_structured))

                newly_scored = [
                    stem for stem in stem_map
                    if stem not in reported_scores and self._has_result_file(predictions_dir, stem)
                ]
                if newly_scored:
                    reported_scores.update(newly_scored)
                    if score_progress_callback is not None:
                        score_progress_callback(len(newly_scored))
                stop_monitor.wait(5.0)

        monitor_thread = threading.Thread(target=_monitor_results, daemon=True)
        if score_progress_callback is not None or structure_progress_callback is not None:
            monitor_thread.start()

        results = []
        command_completed = False
        try:
            run_command(
                [
                    str(self.python), str(self.binary), "predict", str(yaml_dir),
                    "--out_dir", str(out_dir),
                    "--cache", str(self._cfg.paths.boltz_cache_dir),
                    "--accelerator", "gpu",
                    "--devices", "1",
                    "--no_kernels",
                    "--num_workers", "2",
                    "--diffusion_samples_affinity", "1",   # fast: 1 sample per mol
                    "--sampling_steps_affinity", "100",    # fast: 100 steps (vs 200 default)
                ],
                env=env,
                quiet=self._cfg.quiet,
            )
            command_completed = True
            for stem, (orig_idx, smi) in stem_map.items():
                result = self._parse_result(predictions_dir, stem)
                results.append((orig_idx, smi, result))
        finally:
            stop_monitor.set()
            if score_progress_callback is not None or structure_progress_callback is not None:
                monitor_thread.join(timeout=1.0)
                remaining_structures = len(stem_map) - len(reported_structures)
                if command_completed and remaining_structures and structure_progress_callback is not None:
                    structure_progress_callback(remaining_structures)
                remaining_scores = len(stem_map) - len(reported_scores)
                if command_completed and remaining_scores and score_progress_callback is not None:
                    score_progress_callback(remaining_scores)
        return results

    def _get_devices(self) -> list[str]:
        if torch is not None and torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]

    def score(self, smiles_list: list[str]) -> list[dict[str, float | None]]:
        if not smiles_list:
            return []
        self._ensure_cached_msa()

        null_result: dict[str, float | None] = {
            "rank_score": 0.0, "affinity_probability_binary": None, "affinity_pred_value": None,
        }
        results: list[dict[str, float | None] | None] = [None] * len(smiles_list)
        pending: list[tuple[int, str]] = []
        recovered = 0

        for i, smi in enumerate(smiles_list):
            cached = self.cache.get(smi)
            if cached is not None:
                results[i] = cached
                continue
            persisted = self._parse_persistent_prediction(smi)
            if persisted is not None:
                self.cache[smi] = persisted
                results[i] = persisted
                recovered += 1
            else:
                pending.append((i, smi))

        if not pending:
            if recovered:
                self._write_score_cache()
            return [r if r is not None else null_result for r in results]

        if recovered:
            self._write_score_cache()
            print(f"Recovered {recovered} Boltz scores from persistent cache.", flush=True)

        self._ensure_boltz_runtime_cache()
        devices = self._get_devices()
        n_gpus = len(devices)

        # Split pending molecules deterministically across GPUs so each SMILES
        # returns to the same Boltz output shard on restart.
        chunks: list[list[tuple[int, str]]] = [[] for _ in range(n_gpus)]
        for item in pending:
            _, smi = item
            shard = int(hashlib.sha256(smi.encode()).hexdigest(), 16) % n_gpus
            chunks[shard].append(item)

        print(
            f"Boltz scoring {len(pending)} ligands across {n_gpus} GPUs "
            f"(~{len(pending) // n_gpus} per GPU, 1 model load per GPU)",
            flush=True,
        )

        self.scoring_run_dir.mkdir(parents=True, exist_ok=True)
        base = self.scoring_run_dir

        progress_lock = threading.Lock()
        structure_bar = tqdm(
            total=len(smiles_list),
            initial=len(smiles_list) - len(pending),
            desc="Boltz structures",
            unit="ligand",
            dynamic_ncols=True,
            position=0,
        )
        score_bar = tqdm(
            total=len(smiles_list),
            initial=len(smiles_list) - len(pending),
            desc="Boltz scored ligands",
            unit="ligand",
            dynamic_ncols=True,
            position=1,
        )

        def _record_scored(delta: int) -> None:
            if delta <= 0:
                return
            with progress_lock:
                score_bar.update(delta)

        def _record_structured(delta: int) -> None:
            if delta <= 0:
                return
            with progress_lock:
                structure_bar.update(delta)

        def _run_gpu(gpu_idx: int) -> list[tuple[int, str, dict]]:
            chunk = chunks[gpu_idx]
            if not chunk:
                return []
            work_dir = base / f"gpu_{gpu_idx}"
            work_dir.mkdir(parents=True, exist_ok=True)
            return self._score_chunk(
                chunk,
                devices[gpu_idx],
                work_dir,
                score_progress_callback=_record_scored,
                structure_progress_callback=_record_structured,
            )

        scored = 0
        try:
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                gpu_futures = {executor.submit(_run_gpu, i): i for i in range(n_gpus)}
                for future in as_completed(gpu_futures):
                    gpu_idx = gpu_futures[future]
                    try:
                        for orig_idx, smi, result in future.result():
                            results[orig_idx] = result
                            self.cache[smi] = result
                            scored += 1
                        self._write_score_cache()
                    except Exception as exc:
                        print(f"Boltz GPU {devices[gpu_idx]} chunk failed: {exc}", flush=True)
        finally:
            score_bar.close()
            structure_bar.close()

        print(f"Boltz scoring complete: {scored}/{len(pending)} succeeded.", flush=True)
        self._write_score_cache()
        return [r if r is not None else null_result for r in results]


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


def _canonical_smiles_from_mol(mol: Chem.Mol) -> str | None:
    try:
        clean = Chem.Mol(mol)
        try:
            clean = Chem.RemoveHs(clean)
        except Exception:
            pass
        return Chem.MolToSmiles(clean) or None
    except Exception:
        return None


def _first_mol_from_sdf(path: Path) -> Chem.Mol | None:
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        for mol in supplier:
            if mol is not None:
                return mol
    except Exception:
        return None
    return None


def _pose_id_for(*parts: object) -> str:
    key = "|".join(str(p) for p in parts)
    return f"pose_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _allowed_pose_keys(generated_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if generated_df.empty:
        return set()
    return {
        (str(r.generator), str(r.pocket_id), str(r.smiles))
        for r in generated_df[["generator", "pocket_id", "smiles"]].itertuples(index=False)
    }


def _latest_gen_info_by_pocket(generator_root: Path) -> dict[str, Path]:
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


def _find_pxm_pose_sdf(csv_path: Path, filename: str) -> Path | None:
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


def _collect_pocketxmol_pose_records(
    run_dir: Path,
    generated_df: pd.DataFrame,
    generator_name: str,
    allowed: set[tuple[str, str, str]],
) -> list[tuple[PoseRecord, Chem.Mol]]:
    rows: list[tuple[PoseRecord, Chem.Mol]] = []
    root = run_dir / "generator_outputs" / generator_name.lower()
    if not root.exists():
        return rows

    for pocket_id, csv_path in sorted(_latest_gen_info_by_pocket(root).items()):
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
            sdf_path = _find_pxm_pose_sdf(csv_path, filename)
            if sdf_path is None or "-bad" in sdf_path.stem:
                continue

            mol = _first_mol_from_sdf(sdf_path)
            if mol is None:
                continue
            smi = str(row.get("smiles") or "").strip()
            if not smi or not ligand_properties(smi):
                smi = _canonical_smiles_from_mol(mol) or ""
            if not smi or not ligand_properties(smi):
                continue
            if (generator_name, pocket_id, smi) not in allowed:
                continue

            pose_id = _pose_id_for(generator_name, pocket_id, csv_path.parent.name, filename, row_idx)
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


def _collect_diffsbdd_pose_records(
    run_dir: Path,
    generated_df: pd.DataFrame,
    allowed: set[tuple[str, str, str]],
) -> list[tuple[PoseRecord, Chem.Mol]]:
    rows: list[tuple[PoseRecord, Chem.Mol]] = []
    root = run_dir / "generator_outputs" / "diffsbdd"
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
            smi = _canonical_smiles_from_mol(mol) or ""
            if not smi or not ligand_properties(smi):
                continue
            if ("DiffSBDD", pocket_id, smi) not in allowed:
                continue
            pose_id = _pose_id_for("DiffSBDD", pocket_id, sdf_path, mol_idx)
            rows.append((
                PoseRecord(
                    pose_id=pose_id,
                    smiles=smi,
                    generator="DiffSBDD",
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
    allowed = _allowed_pose_keys(generated_df)
    pose_mols: list[tuple[PoseRecord, Chem.Mol]] = []
    pose_mols.extend(_collect_diffsbdd_pose_records(run_dir, generated_df, allowed))
    pose_mols.extend(_collect_pocketxmol_pose_records(run_dir, generated_df, "PocketXMol", allowed))
    pose_mols.extend(_collect_pocketxmol_pose_records(run_dir, generated_df, "PocketXMolAR", allowed))

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

    def _get_devices(self) -> list[str]:
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
        devices = self._get_devices()
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
    else:
        oracle = RTMScorePoseOracle(cfg)
        pose_scores = oracle.score(
            pose_manifest=pose_manifest,
            ligand_sdfs=ligand_sdfs,
            pocket_specs=pocket_specs,
            score_dir=score_dir,
        )
        pose_scores.to_csv(pose_scores_path, index=False)

    pose_level = pose_manifest.merge(pose_scores, on=["pose_id", "pocket_id"], how="left")
    pose_level.to_csv(score_dir / "rtmscore_pose_level.csv", index=False)

    merged = aggregate_rtmscore_scores(unique_df, pose_manifest, pose_scores)
    merged["scorer"] = "rtmscore"
    return merged


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

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
    if generated_df.empty:
        return generated_df.copy()
    return generated_df.groupby("smiles", sort=False).agg(
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


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_scaffold_diversity_matrix(generated_df: pd.DataFrame) -> pd.DataFrame:
    generators = sorted(generated_df["generator"].unique())
    scaffold_sets = {
        g: set(generated_df.loc[(generated_df["generator"] == g) & (generated_df["scaffold"] != ""), "scaffold"])
        for g in generators
    }
    mat = {}
    for g1 in generators:
        mat[g1] = {}
        for g2 in generators:
            union = scaffold_sets[g1] | scaffold_sets[g2]
            inter = scaffold_sets[g1] & scaffold_sets[g2]
            mat[g1][g2] = len(inter) / len(union) if union else 0.0
    return pd.DataFrame(mat, index=generators)


def pocket_jaccard_matrix(pocket_df: pd.DataFrame) -> pd.DataFrame:
    generators = sorted(pocket_df["generator"].unique())
    scaffold_sets = {
        g: set(pocket_df.loc[(pocket_df["generator"] == g) & (pocket_df["scaffold"] != ""), "scaffold"])
        for g in generators
    }
    mat = {}
    for g1 in generators:
        mat[g1] = {}
        for g2 in generators:
            union = scaffold_sets[g1] | scaffold_sets[g2]
            inter = scaffold_sets[g1] & scaffold_sets[g2]
            mat[g1][g2] = len(inter) / len(union) if union else 0.0
    return pd.DataFrame(mat, index=generators)


def pocket_sensitivity(generated_df: pd.DataFrame) -> pd.DataFrame:
    return generated_df.groupby(["generator", "pocket_id"])["smiles"].nunique().rename("n_unique")


def compute_budget_summary(generated_df: pd.DataFrame) -> pd.DataFrame:
    return generated_df.groupby("generator").agg(
        total_mols=("smiles", "count"),
        unique_mols=("smiles", "nunique"),
        unique_scaffolds=("scaffold", lambda s: (s != "").sum()),
        mean_qed=("qed", "mean"),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _ensure_plot_deps() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed.")


def generate_all_figures(
    generated_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    top_hits: pd.DataFrame,
    pocket_specs: list[PocketSpec],
    palette: dict[str, str],
    cfg: Config,
    out_dir: Path,
    scored_df: pd.DataFrame | None = None,
) -> None:
    _ensure_plot_deps()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pocket_dir = out_dir / "per_pocket"
    per_pocket_dir.mkdir(exist_ok=True)

    _save_top_hits_grid(top_hits, out_dir / "top_hits_grid.png")
    _save_ranked_hits(top_hits, out_dir / "ranked_top_hits.png")
    _save_ligand_space(unique_df, palette, cfg, out_dir / "ligand_space_pca.png",
                       generated_df=generated_df)
    _save_ligand_space_umap(unique_df, palette, cfg, out_dir / "ligand_space_umap.png")
    _save_source_pocket_ligand_space(
        generated_df, unique_df, palette, cfg, out_dir / "source_pocket_ligand_space_pca.png"
    )
    _save_source_pocket_ligand_space_umap(
        generated_df, unique_df, palette, cfg, out_dir / "source_pocket_ligand_space_umap.png"
    )
    _save_pocket_tanimoto_landscape(unique_df, cfg, out_dir / "pocket_tanimoto_landscape.png")
    _save_summary_dashboard(generated_df, unique_df, top_hits, pocket_specs, palette, cfg,
                            out_dir / "summary_dashboard.png")
    _save_pocket_generator_heatmap(generated_df, out_dir / "pocket_generator_heatmap.png")
    _save_yield_matrix(generated_df, out_dir / "yield_matrix.png")
    _save_pocket_druggability(unique_df, pocket_specs, out_dir / "pocket_druggability.png")

    # Scaffold family analysis
    if not unique_df.empty and "scaffold" in unique_df.columns:
        try:
            clustered = _cluster_scaffolds(
                unique_df,
                max_points=cfg.max_cluster_points,
                seed=cfg.seed,
            )
            _save_scaffold_family_space(
                clustered, cfg, out_dir / "scaffold_family_space.png"
            )
            _save_scaffold_family_pocket_heatmap(
                clustered, pocket_specs, out_dir / "scaffold_family_pocket_heatmap.png"
            )
        except Exception as exc:
            print(f"WARNING: Scaffold family analysis failed: {exc}")

    ranking_col = (
        "rank_score"
        if "rank_score" in unique_df.columns and unique_df["rank_score"].notna().any()
        else None
    )
    for spec in pocket_specs:
        pocket_mols = unique_df[unique_df["pocket_ids"].str.contains(spec.pocket_id, na=False)]
        if not pocket_mols.empty and ranking_col:
            pocket_top = pocket_mols.sort_values(ranking_col, ascending=False).head(4)
            _save_top_hits_grid(pocket_top, per_pocket_dir / f"{spec.pocket_id}_top_hits.png")


def _write_rdkit_image(image, out_path: Path) -> None:
    if hasattr(image, "save"):
        image.save(str(out_path))
    elif isinstance(image, (bytes, bytearray)):
        out_path.write_bytes(bytes(image))
    elif hasattr(image, "data"):
        out_path.write_bytes(image.data)


def _save_top_hits_grid(top_hits: pd.DataFrame, out_path: Path) -> None:
    _ensure_plot_deps()
    if top_hits.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No top hits available", ha="center", va="center")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    mols, legends = [], []
    for _, row in top_hits.iterrows():
        mol = mol_from_smiles(row["smiles"])
        if mol is None:
            continue
        mols.append(mol)
        gen = row.get("generators", "")
        if pd.notna(row.get("affinity_probability_binary")):
            legends.append(f"{gen}\nPbind={row['affinity_probability_binary']:.2f}")
        else:
            legends.append(f"{gen}\nScore={row.get('rank_score', 0):.2f}")
    if mols:
        image = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=4,
                                     subImgSize=(320, 240), useSVG=False)
        _write_rdkit_image(image, out_path)


def _save_ranked_hits(top_hits: pd.DataFrame, out_path: Path) -> None:
    _ensure_plot_deps()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not top_hits.empty and "rank_score" in top_hits.columns:
        labels = [f"Hit {i + 1}" for i in range(len(top_hits))]
        scores = top_hits["rank_score"].fillna(0).astype(float)
        ax.bar(labels, scores, color="#4E79A7")
        ax.set_ylabel("Ranking score")
        ax.set_title("Top ranked molecules")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No scored hits available", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_ligand_space(
    unique_df: pd.DataFrame, palette: dict[str, str], cfg: Config, out_path: Path,
    generated_df: pd.DataFrame | None = None,
) -> None:
    """Aggregated unique-molecule PCA colored by generator, recurrence/best pocket, and score."""
    _ensure_plot_deps()
    if PCA is None:
        return

    if unique_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    # Compute PCA once on a sample of unique molecules
    plot_df = unique_df.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if not fps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()
    has_best_pocket = "rtmscore_best_pocket_id" in embed_df.columns
    has_recurrence = "n_pockets" in embed_df.columns

    n_panels = 1 + int(has_best_pocket or has_recurrence) + int(has_score)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: colored by generator
    ax = axes[0]
    for gen in sorted(embed_df["generators"].dropna().unique()):
        sub = embed_df[embed_df["generators"] == gen]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.6, c=palette.get(gen, "#4E79A7"), label=gen)
    ax.set_title("Colored by generator")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)

    panel = 1

    # Panel 2: this is an aggregated unique-molecule view, so do not color by
    # comma-joined source pockets. Use best-scoring pose pocket when available,
    # otherwise show recurrence across source pockets as a continuous value.
    if has_best_pocket:
        ax = axes[panel]; panel += 1
        pocket_ids = sorted(embed_df["rtmscore_best_pocket_id"].dropna().unique())
        pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pocket_ids), 1)))
        pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pocket_ids)}
        for pid in pocket_ids:
            sub = embed_df[embed_df["rtmscore_best_pocket_id"] == pid]
            ax.scatter(sub["x"], sub["y"], s=18, alpha=0.6,
                       color=pocket_palette[pid], label=pid)
        ax.set_title("Best-scoring pose pocket")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=8)
    elif has_recurrence:
        ax = axes[panel]; panel += 1
        recurrence = pd.to_numeric(embed_df["n_pockets"], errors="coerce").fillna(1)
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7,
                        c=recurrence, cmap="viridis")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Source pockets rediscovering SMILES")
        ax.set_title("Rediscovery across source pockets")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Panel 3: colored by ranking score (continuous colormap)
    if has_score:
        ax = axes[panel]
        scores = embed_df["rank_score"].fillna(0).astype(float)
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7,
                        c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
        ax.set_title("Colored by ranking score")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle("Ligand chemical space (PCA on Morgan fingerprints)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_ligand_space_umap(
    unique_df: pd.DataFrame, palette: dict[str, str], cfg: Config, out_path: Path,
) -> None:
    """Three-panel ligand space UMAP on Morgan fingerprints, if umap-learn is installed."""
    _ensure_plot_deps()
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        print("WARNING: umap-learn is not installed; skipping ligand_space_umap.png", flush=True)
        return

    if unique_df.empty:
        return

    plot_df = unique_df.copy()
    if len(plot_df) > cfg.max_umap_points:
        plot_df = plot_df.sample(cfg.max_umap_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(f"Building ligand-space UMAP on {len(fps)} sampled molecules ...", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        metric="jaccard",
        random_state=cfg.seed,
        n_neighbors=30,
        min_dist=0.1,
    )
    coords = reducer.fit_transform(np.stack(fps).astype(bool))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()
    has_best_pocket = "rtmscore_best_pocket_id" in embed_df.columns
    has_recurrence = "n_pockets" in embed_df.columns
    n_panels = 1 + int(has_best_pocket or has_recurrence) + int(has_score)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    for gen in sorted(embed_df["generators"].dropna().unique()):
        sub = embed_df[embed_df["generators"] == gen]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.6, c=palette.get(gen, "#4E79A7"), label=gen)
    ax.set_title("Colored by generator")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, fontsize=8)

    panel = 1
    if has_best_pocket:
        ax = axes[panel]; panel += 1
        pocket_ids = sorted(embed_df["rtmscore_best_pocket_id"].dropna().unique())
        pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pocket_ids), 1)))
        pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pocket_ids)}
        for pid in pocket_ids:
            sub = embed_df[embed_df["rtmscore_best_pocket_id"] == pid]
            ax.scatter(sub["x"], sub["y"], s=18, alpha=0.6, color=pocket_palette[pid], label=pid)
        ax.set_title("Best-scoring pose pocket")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.legend(frameon=False, fontsize=8)
    elif has_recurrence:
        ax = axes[panel]; panel += 1
        recurrence = pd.to_numeric(embed_df["n_pockets"], errors="coerce").fillna(1)
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7, c=recurrence, cmap="viridis")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Source pockets rediscovering SMILES")
        ax.set_title("Rediscovery across source pockets")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    if has_score:
        ax = axes[panel]
        scores = embed_df["rank_score"].fillna(0).astype(float)
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
        ax.set_title("Colored by ranking score")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    fig.suptitle("Ligand chemical space (UMAP on Morgan fingerprints)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _scored_source_dataframe(generated_df: pd.DataFrame, unique_df: pd.DataFrame) -> pd.DataFrame:
    if generated_df.empty or unique_df.empty or "smiles" not in generated_df.columns:
        return generated_df.copy()
    score_cols = [
        col for col in [
            "smiles", "rank_score", "scorer", "rtmscore_score", "rtmscore_mean_score",
            "rtmscore_best_pocket_id", "n_pockets", "n_generators",
            "pocket_tanimoto_entropy", "pocket_tanimoto_z_entropy",
            "pocket_tanimoto_z_specificity", "pocket_tanimoto_z_dominant_pocket",
        ]
        if col in unique_df.columns
    ]
    if len(score_cols) <= 1:
        return generated_df.copy()
    return generated_df.merge(unique_df[score_cols].drop_duplicates("smiles"), on="smiles", how="left")


def _save_source_pocket_ligand_space(
    generated_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    palette: dict[str, str],
    cfg: Config,
    out_path: Path,
) -> None:
    """Source-conditioned PCA: one row per generator-pocket molecule, colored by true source pocket."""
    _ensure_plot_deps()
    if PCA is None or generated_df.empty:
        return

    source_df = _scored_source_dataframe(generated_df, unique_df)
    plot_df = source_df.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]
    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()

    fig, axes = plt.subplots(1, 3 if has_score else 2, figsize=(24 if has_score else 16, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    for gen in sorted(embed_df["generator"].dropna().unique()):
        sub = embed_df[embed_df["generator"] == gen]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.55, c=palette.get(gen, "#4E79A7"), label=gen)
    ax.set_title("Source rows colored by generator")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    pocket_ids = sorted(embed_df["pocket_id"].dropna().unique())
    pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pocket_ids), 1)))
    pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pocket_ids)}
    for pid in pocket_ids:
        sub = embed_df[embed_df["pocket_id"] == pid]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.62, color=pocket_palette[pid], label=pid)
    ax.set_title("True conditioning pocket")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)

    if has_score:
        ax = axes[2]
        scores = pd.to_numeric(embed_df["rank_score"], errors="coerce")
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.68, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
        ax.set_title("Unique-molecule score projected onto source rows")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle("Source-conditioned ligand chemical space (PCA on Morgan fingerprints)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_source_pocket_ligand_space_umap(
    generated_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    palette: dict[str, str],
    cfg: Config,
    out_path: Path,
) -> None:
    _ensure_plot_deps()
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        return
    if generated_df.empty:
        return

    source_df = _scored_source_dataframe(generated_df, unique_df)
    plot_df = source_df.copy()
    if len(plot_df) > cfg.max_umap_points:
        plot_df = plot_df.sample(cfg.max_umap_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(f"Building source-pocket UMAP on {len(fps)} sampled source rows ...", flush=True)
    coords = umap.UMAP(
        n_components=2,
        metric="jaccard",
        random_state=cfg.seed,
        n_neighbors=30,
        min_dist=0.1,
    ).fit_transform(np.stack(fps).astype(bool))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]
    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()

    fig, axes = plt.subplots(1, 3 if has_score else 2, figsize=(24 if has_score else 16, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    for gen in sorted(embed_df["generator"].dropna().unique()):
        sub = embed_df[embed_df["generator"] == gen]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.58, c=palette.get(gen, "#4E79A7"), label=gen)
    ax.set_title("Source rows colored by generator")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    pocket_ids = sorted(embed_df["pocket_id"].dropna().unique())
    pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pocket_ids), 1)))
    pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pocket_ids)}
    for pid in pocket_ids:
        sub = embed_df[embed_df["pocket_id"] == pid]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.65, color=pocket_palette[pid], label=pid)
    ax.set_title("True conditioning pocket")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, fontsize=8)

    if has_score:
        ax = axes[2]
        scores = pd.to_numeric(embed_df["rank_score"], errors="coerce")
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
        ax.set_title("Unique-molecule score projected onto source rows")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    fig.suptitle("Source-conditioned ligand chemical space (UMAP on Morgan fingerprints)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_pocket_tanimoto_landscape(
    unique_df: pd.DataFrame, cfg: Config, out_path: Path,
) -> None:
    """PCA landscape colored by pocket-neighborhood Tanimoto specificity."""
    _ensure_plot_deps()
    needed = {"smiles", "pocket_tanimoto_entropy", "pocket_tanimoto_dominant_pocket"}
    if PCA is None or unique_df.empty or not needed.issubset(unique_df.columns):
        return

    plot_df = unique_df.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    ax = axes[0]
    dominant_col = (
        "pocket_tanimoto_z_dominant_pocket"
        if "pocket_tanimoto_z_dominant_pocket" in embed_df.columns
        else "pocket_tanimoto_dominant_pocket"
    )
    entropy_col = (
        "pocket_tanimoto_z_entropy"
        if "pocket_tanimoto_z_entropy" in embed_df.columns
        else "pocket_tanimoto_entropy"
    )
    margin_col = (
        "pocket_tanimoto_z_margin"
        if "pocket_tanimoto_z_margin" in embed_df.columns
        else "pocket_tanimoto_margin"
    )

    pockets = sorted(p for p in embed_df[dominant_col].dropna().unique() if str(p))
    pocket_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pockets), 1)))
    pocket_palette = {pid: pocket_colors[i] for i, pid in enumerate(pockets)}
    for pid in pockets:
        sub = embed_df[embed_df[dominant_col] == pid]
        ax.scatter(sub["x"], sub["y"], s=18, alpha=0.62, color=pocket_palette[pid], label=pid)
    ax.set_title("Nearest pocket chemical neighborhood")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    entropy = pd.to_numeric(embed_df[entropy_col], errors="coerce")
    sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.72, c=entropy, cmap="viridis_r", vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Normalized Tanimoto entropy")
    ax.set_title("Low normalized entropy = pocket-specific")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax = axes[2]
    margin = pd.to_numeric(embed_df.get(margin_col, 0.0), errors="coerce").fillna(0.0)
    sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.72, c=margin, cmap="magma")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Best minus second-best pocket score")
    ax.set_title("Pocket-neighborhood specificity margin")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle("Pocket-specific chemical neighborhoods (Morgan fingerprint PCA)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_chemical_cluster_umap(
    clustered: pd.DataFrame, cfg: Config, out_path: Path,
) -> None:
    """UMAP view of Morgan fingerprint chemical-space basins from HDBSCAN."""
    _ensure_plot_deps()
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        return
    if clustered.empty or "chemical_cluster" not in clustered.columns:
        return

    plot_df = clustered.copy()
    if len(plot_df) > cfg.max_umap_points:
        plot_df = plot_df.sample(cfg.max_umap_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(f"Building chemical-cluster UMAP on {len(fps)} sampled molecules ...", flush=True)
    coords = umap.UMAP(
        n_components=2,
        metric="jaccard",
        random_state=cfg.seed,
        n_neighbors=30,
        min_dist=0.1,
    ).fit_transform(np.stack(fps).astype(bool))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    clusters = sorted(int(c) for c in embed_df["chemical_cluster"].dropna().unique())
    real_clusters = [c for c in clusters if c != -1]
    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()
    fig, axes = plt.subplots(1, 1 + int(has_score), figsize=(9 * (1 + int(has_score)), 8))
    if not has_score:
        axes = [axes]

    ax = axes[0]
    noise = embed_df[embed_df["chemical_cluster"] == -1]
    if not noise.empty:
        ax.scatter(noise["x"], noise["y"], s=10, alpha=0.18, color="lightgrey", label=f"unclustered (n={len(noise)})")
    cmap = plt.cm.tab20 if len(real_clusters) <= 20 else plt.cm.hsv
    for i, cluster_id in enumerate(real_clusters):
        sub = embed_df[embed_df["chemical_cluster"] == cluster_id]
        ax.scatter(
            sub["x"], sub["y"], s=22, alpha=0.75,
            color=cmap(i / max(len(real_clusters), 1)),
            label=f"C{cluster_id} (n={len(sub)})",
        )
    ax.set_title(f"Morgan fingerprint basins (HDBSCAN, {len(real_clusters)} clusters)")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    if len(real_clusters) <= 15:
        ax.legend(frameon=False, fontsize=7, ncol=2)

    if has_score:
        ax2 = axes[1]
        scores = embed_df["rank_score"].fillna(0).astype(float)
        sc = ax2.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.7, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax2, shrink=0.8, label="Ranking score")
        ax2.set_title("Basins colored by ranking score")
        ax2.set_xlabel("UMAP1"); ax2.set_ylabel("UMAP2")

    fig.suptitle("Chemical-space basins from Morgan fingerprints", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scaffold family clustering + figures
# ---------------------------------------------------------------------------

def _cluster_scaffolds(
    unique_df: pd.DataFrame,
    min_cluster_size: int = 8,
    max_points: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Cluster molecules into scaffold families using HDBSCAN on scaffold Morgan FPs.

    Molecules sharing a Bemis-Murcko scaffold cluster are structurally related —
    this gives 'scaffold families' even when exact scaffolds differ.
    Returns unique_df with a new 'scaffold_family' column (int, -1 = noise/singleton).
    """
    import hdbscan

    out = unique_df.copy()
    out["scaffold_family"] = -1

    scaffold_col = out["scaffold"].fillna("")
    # Build per-scaffold fingerprints (cluster scaffolds, propagate to molecules)
    scaffold_counts = scaffold_col[scaffold_col != ""].value_counts()
    unique_scaffolds = scaffold_counts.index.tolist()
    if max_points is not None and len(unique_scaffolds) > max_points:
        unique_scaffolds = unique_scaffolds[:max_points]
        print(
            f"Clustering scaffold families on {len(unique_scaffolds)}/"
            f"{len(scaffold_counts)} most frequent scaffolds ...",
            flush=True,
        )
    else:
        print(f"Clustering scaffold families on {len(unique_scaffolds)} scaffolds ...", flush=True)
    if len(unique_scaffolds) < 2:
        return out

    fps = []
    valid_scaffolds = []
    for sc in unique_scaffolds:
        fp = fp_array(sc)
        if fp is not None:
            fps.append(fp)
            valid_scaffolds.append(sc)

    if len(fps) < 2:
        return out

    X = np.stack(fps)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="jaccard",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    scaffold_to_family = dict(zip(valid_scaffolds, labels.tolist()))
    out["scaffold_family"] = scaffold_col.map(scaffold_to_family).fillna(-1).astype(int)
    return out


def _cluster_molecule_fingerprints(
    unique_df: pd.DataFrame,
    min_cluster_size: int = 25,
    max_points: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster molecules into Morgan fingerprint chemical-space basins."""
    import hdbscan

    out = unique_df.copy()
    out["chemical_cluster"] = -1
    if out.empty:
        return out

    cluster_df = out
    if max_points is not None and len(cluster_df) > max_points:
        cluster_df = cluster_df.sample(max_points, random_state=seed)
        print(
            f"Clustering Morgan fingerprints for presentation CSVs "
            f"on {len(cluster_df)}/{len(out)} sampled molecules ...",
            flush=True,
        )
    else:
        print(
            f"Clustering Morgan fingerprints for presentation CSVs "
            f"on {len(cluster_df)} molecules ...",
            flush=True,
        )

    fps, indices = [], []
    for idx, row in cluster_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            indices.append(idx)
    if len(fps) < max(2, min_cluster_size):
        return out

    X = np.stack(fps).astype(float)
    if PCA is not None and len(fps) > 3:
        n_components = min(50, X.shape[1], len(fps) - 1)
        X_cluster = PCA(n_components=n_components, random_state=seed).fit_transform(X)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(X_cluster)
    else:
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="jaccard",
            cluster_selection_method="eom",
        ).fit_predict(X.astype(bool))
    out.loc[indices, "chemical_cluster"] = labels.astype(int)
    return out


def _split_multi_value(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _composition_string(values: pd.Series, limit: int = 4) -> str:
    from collections import Counter

    counts: Counter[str] = Counter()
    for value in values.dropna():
        for part in _split_multi_value(value):
            counts[part] += 1
    return "; ".join(f"{key}:{count}" for key, count in counts.most_common(limit))


def _write_cluster_summary(clustered: pd.DataFrame, out_path: Path) -> None:
    if clustered.empty or "chemical_cluster" not in clustered.columns:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    rows = []
    score_col = "rank_score" if "rank_score" in clustered.columns else None
    for cluster_id, sub in clustered.groupby("chemical_cluster", sort=True):
        if int(cluster_id) == -1:
            continue
        scaffold = ""
        if "scaffold" in sub.columns and sub["scaffold"].notna().any():
            modes = sub["scaffold"].dropna().astype(str)
            scaffold = modes.mode().iloc[0] if not modes.empty else ""
        row = {
            "cluster_id": int(cluster_id),
            "n_molecules": int(len(sub)),
            "top_scaffold": scaffold,
            "generator_composition": _composition_string(sub.get("generators", pd.Series(dtype=str))),
            "pocket_composition": _composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
            "mean_qed": float(sub["qed"].mean()) if "qed" in sub.columns else np.nan,
            "mean_mw": float(sub["mw"].mean()) if "mw" in sub.columns else np.nan,
            "mean_logp": float(sub["logp"].mean()) if "logp" in sub.columns else np.nan,
        }
        if score_col:
            scores = pd.to_numeric(sub[score_col], errors="coerce")
            row.update({
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "max_score": float(scores.max()),
            })
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame.to_csv(out_path, index=False)
        return
    frame.sort_values(
        ["max_score", "mean_score", "n_molecules"],
        ascending=[False, False, False],
        na_position="last",
    ).to_csv(out_path, index=False)


def _write_scaffold_family_summary(run_dir: Path, merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    try:
        clustered = _cluster_scaffolds(
            merged,
            max_points=cfg.max_cluster_points,
            seed=cfg.seed,
        )
        rows = []
        assigned = clustered[clustered["scaffold_family"] != -1].copy()
        for family_id, sub in assigned.groupby("scaffold_family", sort=True):
            scaffolds = sub["scaffold"].dropna().astype(str) if "scaffold" in sub.columns else pd.Series(dtype=str)
            top_scaffold = scaffolds.mode().iloc[0] if not scaffolds.empty else ""
            scores = pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce")
            rows.append({
                "scaffold_family": int(family_id),
                "n_molecules": int(len(sub)),
                "n_scaffolds": int(scaffolds.nunique()) if not scaffolds.empty else 0,
                "top_scaffold": top_scaffold,
                "generator_composition": _composition_string(sub.get("generators", pd.Series(dtype=str))),
                "pocket_composition": _composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "max_score": float(scores.max()),
                "mean_qed": float(pd.to_numeric(sub.get("qed", np.nan), errors="coerce").mean()),
            })
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(
                ["max_score", "mean_score", "n_molecules"],
                ascending=[False, False, False],
                na_position="last",
            )
        frame.to_csv(run_dir / "scaffold_family_summary.csv", index=False)
        return clustered
    except Exception as exc:
        print(f"WARNING: scaffold family summary failed: {exc}", flush=True)
        pd.DataFrame().to_csv(run_dir / "scaffold_family_summary.csv", index=False)
        return merged.copy()


def _write_top_hit_enrichment(
    df: pd.DataFrame,
    label_col: str,
    out_path: Path,
    label_name: str,
    score_col: str = "rank_score",
) -> None:
    if df.empty or label_col not in df.columns or score_col not in df.columns:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    records = []
    base = df[["smiles", label_col, score_col]].copy()
    base[score_col] = pd.to_numeric(base[score_col], errors="coerce").fillna(0.0)
    exploded_rows = []
    for _, row in base.iterrows():
        if label_col == "chemical_cluster":
            if pd.isna(row[label_col]):
                labels = []
            else:
                labels = [str(int(row[label_col]))]
        else:
            labels = _split_multi_value(row[label_col])
        for label in labels:
            if label and label != "-1":
                exploded_rows.append({"smiles": row["smiles"], label_name: label, score_col: row[score_col]})
    exploded = pd.DataFrame(exploded_rows)
    if exploded.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    totals = exploded.groupby(label_name)["smiles"].nunique()
    ranked = base.sort_values(score_col, ascending=False)
    for frac in (0.01, 0.05, 0.10):
        n_top = max(1, int(np.ceil(len(ranked) * frac)))
        top_smiles = set(ranked.head(n_top)["smiles"])
        top = exploded[exploded["smiles"].isin(top_smiles)]
        top_counts = top.groupby(label_name)["smiles"].nunique()
        for label, total in totals.items():
            n_label_top = int(top_counts.get(label, 0))
            records.append({
                "top_fraction": frac,
                label_name: label,
                "n_top": n_label_top,
                "n_total": int(total),
                "fraction_of_top_hits": n_label_top / n_top,
                "top_hit_rate_within_label": n_label_top / int(total) if total else 0.0,
            })
    pd.DataFrame(records).sort_values(
        ["top_fraction", "n_top", "top_hit_rate_within_label"],
        ascending=[True, False, False],
    ).to_csv(out_path, index=False)


def _mean_topk_tanimoto(query_fp: Any, ref_fps: list[Any], top_k: int) -> float:
    if not ref_fps:
        return 0.0
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps), dtype=float)
    if sims.size == 0:
        return 0.0
    k = max(1, min(int(top_k), sims.size))
    if sims.size > k:
        sims = np.partition(sims, -k)[-k:]
    return float(np.mean(sims))


def _write_pocket_tanimoto_analysis(
    run_dir: Path,
    generated_df: pd.DataFrame,
    merged: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    out = merged.copy()
    if out.empty or "smiles" not in out.columns or "pocket_id" not in generated_df.columns:
        return out

    pocket_refs: dict[str, list[Any]] = {}
    for pocket_id, sub in generated_df.groupby("pocket_id", sort=True):
        smiles = sub["smiles"].dropna().astype(str).drop_duplicates()
        if cfg.max_tanimoto_refs_per_pocket and len(smiles) > cfg.max_tanimoto_refs_per_pocket:
            smiles = smiles.sample(cfg.max_tanimoto_refs_per_pocket, random_state=cfg.seed)
        fps = [fp for fp in (morgan_fp(smi) for smi in smiles) if fp is not None]
        if fps:
            pocket_refs[str(pocket_id)] = fps

    pocket_ids = sorted(pocket_refs)
    if len(pocket_ids) < 2:
        return out

    print(
        "Computing pocket Tanimoto entropy "
        f"({len(out)} molecules, {len(pocket_ids)} pockets, "
        f"up to {cfg.max_tanimoto_refs_per_pocket} refs/pocket) ...",
        flush=True,
    )

    sim_cols = {pid: f"tanimoto_to_{pid}" for pid in pocket_ids}
    for col in sim_cols.values():
        out[col] = np.nan
    out["pocket_tanimoto_entropy"] = np.nan
    out["pocket_tanimoto_specificity"] = np.nan
    out["pocket_tanimoto_dominant_pocket"] = ""
    out["pocket_tanimoto_max"] = np.nan
    out["pocket_tanimoto_second"] = np.nan
    out["pocket_tanimoto_margin"] = np.nan
    out["pocket_tanimoto_z_entropy"] = np.nan
    out["pocket_tanimoto_z_specificity"] = np.nan
    out["pocket_tanimoto_z_dominant_pocket"] = ""
    out["pocket_tanimoto_z_margin"] = np.nan

    denom = np.log(len(pocket_ids))
    eps = 1e-12
    for idx, row in out.iterrows():
        fp = morgan_fp(str(row["smiles"]))
        if fp is None:
            continue
        sims = np.asarray([
            _mean_topk_tanimoto(fp, pocket_refs[pid], cfg.tanimoto_top_k)
            for pid in pocket_ids
        ], dtype=float)
        for pid, value in zip(pocket_ids, sims):
            out.at[idx, sim_cols[pid]] = value
        order = np.argsort(sims)[::-1]
        best = float(sims[order[0]])
        second = float(sims[order[1]]) if len(order) > 1 else 0.0
        total = float(sims.sum())
        if total > eps:
            probs = sims / total
            entropy = float(-(probs * np.log(probs + eps)).sum() / denom)
        else:
            entropy = np.nan
        out.at[idx, "pocket_tanimoto_entropy"] = entropy
        out.at[idx, "pocket_tanimoto_specificity"] = 1.0 - entropy if pd.notna(entropy) else np.nan
        out.at[idx, "pocket_tanimoto_dominant_pocket"] = pocket_ids[int(order[0])]
        out.at[idx, "pocket_tanimoto_max"] = best
        out.at[idx, "pocket_tanimoto_second"] = second
        out.at[idx, "pocket_tanimoto_margin"] = best - second

    sim_frame = out[list(sim_cols.values())].apply(pd.to_numeric, errors="coerce")
    means = sim_frame.mean(axis=0)
    stds = sim_frame.std(axis=0).replace(0, np.nan)
    z_frame = (sim_frame - means) / stds
    z_frame = z_frame.fillna(0.0)
    for idx, z_row in z_frame.iterrows():
        z = z_row.to_numpy(dtype=float)
        shifted = z - np.max(z)
        weights = np.exp(shifted)
        probs = weights / max(float(weights.sum()), eps)
        order = np.argsort(z)[::-1]
        entropy = float(-(probs * np.log(probs + eps)).sum() / denom)
        best = float(z[order[0]])
        second = float(z[order[1]]) if len(order) > 1 else 0.0
        out.at[idx, "pocket_tanimoto_z_entropy"] = entropy
        out.at[idx, "pocket_tanimoto_z_specificity"] = 1.0 - entropy
        out.at[idx, "pocket_tanimoto_z_dominant_pocket"] = pocket_ids[int(order[0])]
        out.at[idx, "pocket_tanimoto_z_margin"] = best - second

    analysis_cols = [
        "smiles", "generators", "pocket_ids", "rank_score",
        "pocket_tanimoto_dominant_pocket", "pocket_tanimoto_entropy",
        "pocket_tanimoto_specificity", "pocket_tanimoto_max",
        "pocket_tanimoto_second", "pocket_tanimoto_margin",
        "pocket_tanimoto_z_dominant_pocket", "pocket_tanimoto_z_entropy",
        "pocket_tanimoto_z_specificity", "pocket_tanimoto_z_margin",
        *sim_cols.values(),
    ]
    available_cols = [col for col in analysis_cols if col in out.columns]
    out[available_cols].to_csv(run_dir / "pocket_tanimoto_entropy.csv", index=False)

    source_rows = []
    for _, row in out.iterrows():
        for source_pocket in _split_multi_value(row.get("pocket_ids", "")):
            source_rows.append({
                "source_pocket": source_pocket,
                "dominant_tanimoto_pocket": row.get("pocket_tanimoto_dominant_pocket", ""),
                "rank_score": row.get("rank_score", np.nan),
                "entropy": row.get("pocket_tanimoto_entropy", np.nan),
                "specificity": row.get("pocket_tanimoto_specificity", np.nan),
                "margin": row.get("pocket_tanimoto_margin", np.nan),
                "z_dominant_tanimoto_pocket": row.get("pocket_tanimoto_z_dominant_pocket", ""),
                "z_entropy": row.get("pocket_tanimoto_z_entropy", np.nan),
                "z_specificity": row.get("pocket_tanimoto_z_specificity", np.nan),
                "z_margin": row.get("pocket_tanimoto_z_margin", np.nan),
            })
    source_df = pd.DataFrame(source_rows)
    if not source_df.empty:
        summary_rows = []
        for source_pocket, sub in source_df.groupby("source_pocket", sort=True):
            match = sub["dominant_tanimoto_pocket"].astype(str) == str(source_pocket)
            summary_rows.append({
                "source_pocket": source_pocket,
                "n_molecules": int(len(sub)),
                "mean_entropy": float(pd.to_numeric(sub["entropy"], errors="coerce").mean()),
                "mean_specificity": float(pd.to_numeric(sub["specificity"], errors="coerce").mean()),
                "mean_margin": float(pd.to_numeric(sub["margin"], errors="coerce").mean()),
                "mean_z_entropy": float(pd.to_numeric(sub["z_entropy"], errors="coerce").mean()),
                "mean_z_specificity": float(pd.to_numeric(sub["z_specificity"], errors="coerce").mean()),
                "mean_z_margin": float(pd.to_numeric(sub["z_margin"], errors="coerce").mean()),
                "mean_rank_score": float(pd.to_numeric(sub["rank_score"], errors="coerce").mean()),
                "fraction_nearest_to_own_pocket": float(match.mean()),
                "z_fraction_nearest_to_own_pocket": float(
                    (sub["z_dominant_tanimoto_pocket"].astype(str) == str(source_pocket)).mean()
                ),
                "dominant_pocket_composition": _composition_string(sub["dominant_tanimoto_pocket"]),
                "z_dominant_pocket_composition": _composition_string(sub["z_dominant_tanimoto_pocket"]),
            })
        pd.DataFrame(summary_rows).to_csv(run_dir / "pocket_tanimoto_summary.csv", index=False)
        pd.crosstab(
            source_df["source_pocket"],
            source_df["dominant_tanimoto_pocket"],
        ).to_csv(run_dir / "pocket_tanimoto_confusion.csv")

    return out


def _compute_tanimoto_matrix(fps_a: list[Any], fps_b: list[Any]) -> np.ndarray:
    if not fps_a or not fps_b:
        return np.zeros((len(fps_a), len(fps_b)), dtype=float)
    return np.asarray(
        [DataStructs.BulkTanimotoSimilarity(fp, fps_b) for fp in fps_a],
        dtype=float,
    )


def _write_pocket_distribution_metrics(run_dir: Path, generated_df: pd.DataFrame, cfg: Config) -> None:
    if generated_df.empty or "pocket_id" not in generated_df.columns:
        pd.DataFrame().to_csv(run_dir / "pocket_chemical_distribution_metrics.csv", index=False)
        return

    pocket_fps: dict[str, list[Any]] = {}
    for pocket_id, sub in generated_df.groupby("pocket_id", sort=True):
        smiles = sub["smiles"].dropna().astype(str).drop_duplicates()
        if cfg.max_tanimoto_refs_per_pocket and len(smiles) > cfg.max_tanimoto_refs_per_pocket:
            smiles = smiles.sample(cfg.max_tanimoto_refs_per_pocket, random_state=cfg.seed)
        fps = [fp for fp in (morgan_fp(smi) for smi in smiles) if fp is not None]
        if fps:
            pocket_fps[str(pocket_id)] = fps

    rows = []
    for source, fps_a in pocket_fps.items():
        for target, fps_b in pocket_fps.items():
            sims = _compute_tanimoto_matrix(fps_a, fps_b)
            if sims.size == 0:
                continue
            if source == target and sims.shape[0] == sims.shape[1]:
                np.fill_diagonal(sims, np.nan)
            nearest = np.nanmax(sims, axis=1)
            rows.append({
                "source_pocket": source,
                "target_pocket": target,
                "n_source": int(len(fps_a)),
                "n_target": int(len(fps_b)),
                "mean_nearest_tanimoto": float(np.nanmean(nearest)),
                "median_nearest_tanimoto": float(np.nanmedian(nearest)),
                "p90_nearest_tanimoto": float(np.nanpercentile(nearest, 90)),
                "mean_pairwise_tanimoto": float(np.nanmean(sims)),
            })
    pd.DataFrame(rows).to_csv(run_dir / "pocket_chemical_distribution_metrics.csv", index=False)


def _write_score_correlation_metrics(run_dir: Path, merged: pd.DataFrame) -> None:
    if merged.empty or "rank_score" not in merged.columns:
        pd.DataFrame().to_csv(run_dir / "score_correlation_metrics.csv", index=False)
        return
    candidates = [
        "mw", "logp", "qed", "hbd", "hba", "rot_bonds", "rings",
        "n_generators", "n_pockets", "source_min_rank",
        "rtmscore_n_poses", "pocket_tanimoto_entropy",
        "pocket_tanimoto_specificity", "pocket_tanimoto_margin",
        "pocket_tanimoto_z_entropy", "pocket_tanimoto_z_specificity",
        "pocket_tanimoto_z_margin",
    ]
    rows = []
    score = pd.to_numeric(merged["rank_score"], errors="coerce")
    for col in candidates:
        if col not in merged.columns:
            continue
        values = pd.to_numeric(merged[col], errors="coerce")
        valid = score.notna() & values.notna()
        if int(valid.sum()) < 3:
            continue
        rows.append({
            "feature": col,
            "n": int(valid.sum()),
            "pearson_r": float(score[valid].corr(values[valid], method="pearson")),
            "spearman_r": float(score[valid].corr(values[valid], method="spearman")),
        })
    pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).to_csv(
        run_dir / "score_correlation_metrics.csv", index=False
    )


def _write_source_pocket_predictability(run_dir: Path, generated_df: pd.DataFrame, cfg: Config) -> None:
    if generated_df.empty or generated_df["pocket_id"].nunique() < 2:
        pd.DataFrame().to_csv(run_dir / "source_pocket_predictability.csv", index=False)
        return
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except Exception as exc:
        print(f"WARNING: sklearn classifier unavailable; skipping source pocket predictability: {exc}", flush=True)
        pd.DataFrame().to_csv(run_dir / "source_pocket_predictability.csv", index=False)
        return

    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("all_generators", generated_df)]
    groups.extend((str(gen), sub) for gen, sub in generated_df.groupby("generator", sort=True))
    for label, sub in groups:
        sub = sub.dropna(subset=["smiles", "pocket_id"]).copy()
        if len(sub) > cfg.max_pca_points:
            sub = sub.sample(cfg.max_pca_points, random_state=cfg.seed)
        counts = sub["pocket_id"].value_counts()
        if len(counts) < 2 or counts.min() < 3:
            continue
        fps, labels = [], []
        for row in sub.itertuples(index=False):
            arr = fp_array(str(row.smiles))
            if arr is not None:
                fps.append(arr)
                labels.append(str(row.pocket_id))
        if len(set(labels)) < 2 or len(fps) < 6:
            continue
        X = np.stack(fps)
        y = np.asarray(labels)
        min_class = min(np.unique(y, return_counts=True)[1])
        n_splits = max(2, min(5, int(min_class)))
        model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
        rng = np.random.default_rng(cfg.seed)
        shuffled = y.copy()
        rng.shuffle(shuffled)
        null_scores = cross_val_score(model, X, shuffled, cv=cv, scoring="balanced_accuracy")
        rows.append({
            "subset": label,
            "n": int(len(y)),
            "n_pockets": int(len(set(y))),
            "balanced_accuracy_mean": float(np.mean(scores)),
            "balanced_accuracy_std": float(np.std(scores)),
            "shuffled_balanced_accuracy_mean": float(np.mean(null_scores)),
            "delta_vs_shuffled": float(np.mean(scores) - np.mean(null_scores)),
        })
    pd.DataFrame(rows).to_csv(run_dir / "source_pocket_predictability.csv", index=False)


def _butina_ecfp_families(df: pd.DataFrame, cfg: Config, sim_threshold: float | None = None) -> pd.DataFrame:
    from rdkit.ML.Cluster import Butina

    out = df.copy()
    out["ecfp_family"] = -1
    if out.empty:
        return out
    sample = out
    if len(sample) > cfg.max_cluster_points:
        sample = sample.sample(cfg.max_cluster_points, random_state=cfg.seed)
    fps, indices = [], []
    for idx, row in sample.iterrows():
        fp = morgan_fp(str(row["smiles"]))
        if fp is not None:
            fps.append(fp)
            indices.append(idx)
    if len(fps) < 2:
        return out
    distances = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend([1.0 - x for x in sims])
    threshold = cfg.ecfp_family_sim_threshold if sim_threshold is None else sim_threshold
    clusters = Butina.ClusterData(distances, len(fps), 1.0 - threshold, isDistData=True)
    for family_id, cluster in enumerate(clusters):
        for local_idx in cluster:
            out.at[indices[int(local_idx)], "ecfp_family"] = int(family_id)
    return out


def _write_ecfp_family_outputs(run_dir: Path, merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    try:
        families = _butina_ecfp_families(merged, cfg)
        assigned = families[families["ecfp_family"] != -1].copy()
        rows = []
        for family_id, sub in assigned.groupby("ecfp_family", sort=True):
            scaffold = ""
            if "scaffold" in sub.columns:
                scaffolds = sub["scaffold"].dropna().astype(str)
                scaffold = scaffolds.mode().iloc[0] if not scaffolds.empty else ""
            medoid = sub.sort_values("rank_score", ascending=False)["smiles"].iloc[0] if "rank_score" in sub else sub["smiles"].iloc[0]
            rows.append({
                "ecfp_family": int(family_id),
                "n_molecules": int(len(sub)),
                "representative_smiles": medoid,
                "top_scaffold": scaffold,
                "generator_composition": _composition_string(sub.get("generators", pd.Series(dtype=str))),
                "pocket_composition": _composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
                "mean_rank_score": float(pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce").mean()),
                "max_rank_score": float(pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce").max()),
                "mean_qed": float(pd.to_numeric(sub.get("qed", np.nan), errors="coerce").mean()),
            })
        summary = pd.DataFrame(rows).sort_values(
            ["n_molecules", "max_rank_score"], ascending=[False, False], na_position="last"
        )
        families, summary = _assign_ecfp_hierarchical_groups(families, summary)
        summary.to_csv(run_dir / "ecfp_family_summary.csv", index=False)
        _write_ecfp_group_summary(run_dir, families, summary)
        families.to_csv(run_dir / "ecfp_family_assignments.csv", index=False)
        return families
    except Exception as exc:
        print(f"WARNING: ECFP family analysis failed: {exc}", flush=True)
        pd.DataFrame().to_csv(run_dir / "ecfp_family_summary.csv", index=False)
        return merged.copy()


def _assign_ecfp_hierarchical_groups(
    families: pd.DataFrame,
    summary: pd.DataFrame,
    max_groups: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = families.copy()
    summary = summary.copy()
    families["ecfp_group"] = -1
    summary["ecfp_group"] = -1
    if summary.empty or "representative_smiles" not in summary.columns:
        return families, summary

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        for i, family_id in enumerate(summary["ecfp_family"].astype(int).tolist()):
            group_id = i % max_groups
            summary.loc[summary["ecfp_family"].astype(int) == family_id, "ecfp_group"] = group_id
            families.loc[families["ecfp_family"].astype(int) == family_id, "ecfp_group"] = group_id
        return families, summary

    fps, family_ids = [], []
    for row in summary.itertuples(index=False):
        fp = morgan_fp(str(row.representative_smiles))
        if fp is not None:
            fps.append(fp)
            family_ids.append(int(row.ecfp_family))
    if len(fps) < 2:
        if family_ids:
            families.loc[families["ecfp_family"].isin(family_ids), "ecfp_group"] = 0
            summary.loc[summary["ecfp_family"].isin(family_ids), "ecfp_group"] = 0
        return families, summary

    sim = _compute_tanimoto_matrix(fps, fps)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    linked = linkage(squareform(dist, checks=False), method="average")
    n_groups = max(2, min(max_groups, len(family_ids)))
    raw_groups = fcluster(linked, t=n_groups, criterion="maxclust")

    # Stable group ids by the best-scoring/largest family in each group.
    group_members: dict[int, list[int]] = {}
    for family_id, raw_group in zip(family_ids, raw_groups):
        group_members.setdefault(int(raw_group), []).append(family_id)
    group_order = sorted(
        group_members,
        key=lambda gid: (
            -int(summary.loc[summary["ecfp_family"].isin(group_members[gid]), "n_molecules"].sum()),
            -float(summary.loc[summary["ecfp_family"].isin(group_members[gid]), "max_rank_score"].max()),
        ),
    )
    group_map = {raw_group: i for i, raw_group in enumerate(group_order)}
    family_to_group = {
        family_id: group_map[int(raw_group)]
        for family_id, raw_group in zip(family_ids, raw_groups)
    }
    summary["ecfp_group"] = summary["ecfp_family"].astype(int).map(family_to_group).fillna(-1).astype(int)
    families["ecfp_group"] = families["ecfp_family"].astype(int).map(family_to_group).fillna(-1).astype(int)
    return families, summary


def _write_ecfp_group_summary(run_dir: Path, families: pd.DataFrame, summary: pd.DataFrame) -> None:
    if families.empty or "ecfp_group" not in families.columns:
        pd.DataFrame().to_csv(run_dir / "ecfp_group_summary.csv", index=False)
        return
    assigned = families[families["ecfp_group"] != -1].copy()
    rows = []
    for group_id, sub in assigned.groupby("ecfp_group", sort=True):
        fam_sub = summary[summary["ecfp_group"] == group_id] if "ecfp_group" in summary else pd.DataFrame()
        rows.append({
            "ecfp_group": int(group_id),
            "n_molecules": int(len(sub)),
            "n_families": int(sub["ecfp_family"].nunique()) if "ecfp_family" in sub else 0,
            "top_families": "; ".join(
                f"F{int(r.ecfp_family)}:n={int(r.n_molecules)}"
                for r in fam_sub.head(6).itertuples(index=False)
            ) if not fam_sub.empty else "",
            "generator_composition": _composition_string(sub.get("generators", pd.Series(dtype=str))),
            "pocket_composition": _composition_string(sub.get("pocket_ids", pd.Series(dtype=str))),
            "mean_rank_score": float(pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce").mean()),
            "max_rank_score": float(pd.to_numeric(sub.get("rank_score", np.nan), errors="coerce").max()),
        })
    pd.DataFrame(rows).sort_values(
        ["n_molecules", "max_rank_score"], ascending=[False, False], na_position="last"
    ).to_csv(run_dir / "ecfp_group_summary.csv", index=False)


def _save_ecfp_family_landscape(families: pd.DataFrame, cfg: Config, out_path: Path) -> None:
    _ensure_plot_deps()
    if PCA is None or families.empty or "ecfp_family" not in families.columns:
        return
    plot_df = families[families["ecfp_family"] != -1].copy()
    if plot_df.empty:
        return
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)
    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if len(fps) < 3:
        return
    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]
    group_col = "ecfp_group" if "ecfp_group" in embed_df.columns else "ecfp_family"
    groups_sorted = sorted(int(g) for g in embed_df[group_col].dropna().unique() if int(g) != -1)
    group_counts = embed_df[group_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    cmap = plt.cm.tab10 if len(groups_sorted) <= 10 else plt.cm.tab20
    for i, group_id in enumerate(groups_sorted):
        sub = embed_df[embed_df[group_col].astype(int) == group_id]
        n_families = int(sub["ecfp_family"].nunique())
        label = f"G{group_id} ({len(sub)} mols, {n_families} fams)"
        ax.scatter(
            sub["x"], sub["y"], s=20, alpha=0.68,
            color=cmap(i / max(len(groups_sorted) - 1, 1)),
            label=label,
        )
    ax.set_title("Hierarchical ECFP/Tanimoto groups")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=7, ncol=1 if len(groups_sorted) <= 8 else 2)
    ax = axes[1]
    if "rank_score" in embed_df.columns:
        scores = pd.to_numeric(embed_df["rank_score"], errors="coerce")
        sc = ax.scatter(embed_df["x"], embed_df["y"], s=18, alpha=0.72, c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Ranking score")
    ax.set_title("Same projection colored by score")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Hierarchical ECFP chemical modules (sampled Butina families)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_ecfp_family_tree(run_dir: Path, out_path: Path) -> None:
    _ensure_plot_deps()
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        return
    summary_path = run_dir / "ecfp_family_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if summary.empty or "representative_smiles" not in summary.columns:
        return
    top = summary.head(30).copy()
    fps, labels = [], []
    for row in top.itertuples(index=False):
        fp = morgan_fp(str(row.representative_smiles))
        if fp is not None:
            fps.append(fp)
            labels.append(f"F{int(row.ecfp_family)} n={int(row.n_molecules)}")
    if len(fps) < 3:
        return
    sim = _compute_tanimoto_matrix(fps, fps)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    linked = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.28)))
    dendrogram(linked, labels=labels, orientation="right", ax=ax, color_threshold=0.5)
    ax.set_xlabel("1 - Tanimoto similarity")
    ax.set_title("Hierarchy of top ECFP/Tanimoto families")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_score_summary_plot(
    summary_csv: Path,
    out_path: Path,
    id_col: str,
    title: str,
    label_prefix: str,
    max_rows: int = 16,
) -> None:
    _ensure_plot_deps()
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    if "mean_score" not in df.columns and "mean_rank_score" in df.columns:
        df = df.rename(columns={"mean_rank_score": "mean_score"})
    if "max_score" not in df.columns and "max_rank_score" in df.columns:
        df = df.rename(columns={"max_rank_score": "max_score"})
    needed = {id_col, "n_molecules", "mean_score", "max_score"}
    if not needed.issubset(df.columns):
        return
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
    df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
    df["n_molecules"] = pd.to_numeric(df["n_molecules"], errors="coerce").fillna(0).astype(int)
    plot_df = df.dropna(subset=["mean_score", "max_score"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(
        ["max_score", "mean_score", "n_molecules"],
        ascending=[False, False, False],
    ).head(max_rows)
    labels = [f"{label_prefix}{int(x)}\nn={n}" for x, n in zip(plot_df[id_col], plot_df["n_molecules"])]
    x = np.arange(len(plot_df))

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(plot_df) * 0.65), 5.6))
    ax = axes[0]
    ax.bar(x, plot_df["mean_score"], color="#4E79A7", alpha=0.82, label="mean")
    ax.scatter(x, plot_df["max_score"], color="#D62728", s=38, zorder=3, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Ranking score")
    ax.set_title("Score by group")
    ax.legend(frameon=False)

    ax = axes[1]
    size = np.sqrt(plot_df["n_molecules"].clip(lower=1)) * 28
    sc = ax.scatter(
        plot_df["n_molecules"],
        plot_df["mean_score"],
        s=size,
        c=plot_df["max_score"],
        cmap="RdYlGn",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.35,
    )
    for _, row in plot_df.head(10).iterrows():
        ax.text(row["n_molecules"], row["mean_score"], f"{label_prefix}{int(row[id_col])}", fontsize=8)
    ax.set_xlabel("Molecules in group")
    ax.set_ylabel("Mean ranking score")
    ax.set_title("Size vs score")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Max ranking score")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _scaffold_family_label(family_id: int, family_smiles: list[str]) -> str:
    """Generate a short human-readable label for a scaffold family."""
    if family_id == -1:
        return "singleton"
    if not family_smiles:
        return f"F{family_id}"
    # Pick the most common scaffold in the family as the label
    from collections import Counter
    most_common = Counter(family_smiles).most_common(1)[0][0]
    # Truncate long SMILES
    label = most_common[:18] + "…" if len(most_common) > 18 else most_common
    return f"F{family_id}: {label}"


def _save_scaffold_family_space(
    clustered: pd.DataFrame, cfg: Config, out_path: Path,
) -> None:
    """
    PCA of molecules colored by scaffold family.
    Reveals whether the chemical space clusters align with scaffold families,
    and which families are associated with high ranking scores.
    """
    _ensure_plot_deps()
    if PCA is None or clustered.empty:
        return

    plot_df = clustered.copy()
    if len(plot_df) > cfg.max_pca_points:
        plot_df = plot_df.sample(cfg.max_pca_points, random_state=cfg.seed)

    fps, keep_rows = [], []
    for _, row in plot_df.iterrows():
        arr = fp_array(row["smiles"])
        if arr is not None:
            fps.append(arr)
            keep_rows.append(row)
    if not fps:
        return

    embed_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    coords = PCA(n_components=2, random_state=cfg.seed).fit_transform(np.stack(fps))
    embed_df["x"], embed_df["y"] = coords[:, 0], coords[:, 1]

    families = sorted(embed_df["scaffold_family"].unique())
    n_real = len([f for f in families if f != -1])
    cmap = plt.cm.tab20 if n_real <= 20 else plt.cm.hsv

    has_score = "rank_score" in embed_df.columns and embed_df["rank_score"].notna().any()
    fig, axes = plt.subplots(1, 1 + int(has_score), figsize=(9 * (1 + int(has_score)), 8))
    if not has_score:
        axes = [axes]

    ax = axes[0]
    real_families = [f for f in families if f != -1]
    colors = {f: cmap(i / max(len(real_families), 1)) for i, f in enumerate(real_families)}
    colors[-1] = (0.8, 0.8, 0.8, 0.3)  # singletons: light grey, very transparent

    # Plot singletons first (background)
    noise = embed_df[embed_df["scaffold_family"] == -1]
    if not noise.empty:
        ax.scatter(noise["x"], noise["y"], s=12, alpha=0.2, color="lightgrey", label=f"singleton (n={len(noise)})")

    for fam in real_families:
        sub = embed_df[embed_df["scaffold_family"] == fam]
        scaffolds = sub["scaffold"].tolist() if "scaffold" in sub.columns else []
        label = _scaffold_family_label(fam, scaffolds)
        ax.scatter(sub["x"], sub["y"], s=25, alpha=0.75, color=colors[fam], label=label)

    ax.set_title(f"Scaffold families (HDBSCAN, {n_real} families)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if n_real <= 15:
        ax.legend(frameon=False, fontsize=7, ncol=2)

    if has_score:
        ax2 = axes[1]
        scores = embed_df["rank_score"].fillna(0).astype(float)
        span = scores.max() - scores.min()
        normalized = (scores - scores.min()) / span if span > 0 else scores * 0
        sizes = 10 + 40 * normalized
        sc = ax2.scatter(embed_df["x"], embed_df["y"], s=sizes, alpha=0.6,
                         c=scores, cmap="RdYlGn")
        plt.colorbar(sc, ax=ax2, shrink=0.8, label="Ranking score")
        ax2.set_title("Scaffold families sized and colored by score")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

    fig.suptitle("Chemical space colored by scaffold family", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_scaffold_family_pocket_heatmap(
    clustered: pd.DataFrame,
    pocket_specs: list[PocketSpec],
    out_path: Path,
) -> None:
    """
    Heatmap: scaffold family × pocket — mean ranking score.

    Each cell shows the mean ranking score for molecules of that scaffold
    family when generated for that pocket. Reveals pocket-selective scaffold families.
    Also shows n_mols per cell as text annotation.
    """
    _ensure_plot_deps()
    if clustered.empty:
        return

    has_score = "rank_score" in clustered.columns and clustered["rank_score"].notna().any()
    value_col = "rank_score" if has_score else "qed"

    scored = clustered[clustered["scaffold_family"] != -1].copy()
    if scored.empty:
        return

    # Expand pocket_ids (comma-joined) into one row per pocket
    rows = []
    for _, r in scored.iterrows():
        for pid in str(r.get("pocket_ids", "")).split(", "):
            pid = pid.strip()
            if pid:
                rows.append({**r.to_dict(), "pocket_id_single": pid})
    if not rows:
        return
    expanded = pd.DataFrame(rows)

    pivot_mean = expanded.groupby(["scaffold_family", "pocket_id_single"])[value_col].mean().unstack(fill_value=np.nan)
    pivot_count = expanded.groupby(["scaffold_family", "pocket_id_single"])[value_col].count().unstack(fill_value=0)

    # Order families by mean score descending
    family_order = pivot_mean.mean(axis=1).sort_values(ascending=False).index
    pivot_mean = pivot_mean.loc[family_order]
    pivot_count = pivot_count.loc[family_order]

    # Pocket column order matches pocket_specs order
    pocket_order = [s.pocket_id for s in pocket_specs if s.pocket_id in pivot_mean.columns]
    if pocket_order:
        pivot_mean = pivot_mean.reindex(columns=pocket_order, fill_value=np.nan)
        pivot_count = pivot_count.reindex(columns=pocket_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot_mean.columns) * 2),
                                    max(6, len(pivot_mean) * 0.5 + 2)))
    if value_col == "qed":
        im = ax.imshow(pivot_mean.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    else:
        im = ax.imshow(pivot_mean.values.astype(float), aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_xticklabels(pivot_mean.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_mean.index)))

    # Family labels: family_id + example scaffold
    family_labels = []
    for fid in pivot_mean.index:
        sub = clustered[clustered["scaffold_family"] == fid]["scaffold"]
        example = sub.mode().iloc[0] if len(sub) else ""
        example = (example[:15] + "…") if len(example) > 15 else example
        family_labels.append(f"F{fid}: {example}")
    ax.set_yticklabels(family_labels, fontsize=8)

    # Annotate with count
    for i in range(len(pivot_mean.index)):
        for j in range(len(pivot_mean.columns)):
            cnt = int(pivot_count.values[i, j])
            val = pivot_mean.values[i, j]
            if cnt > 0:
                txt = f"{val:.2f}\n(n={cnt})"
                color = "black" if (np.isnan(val) or 0.3 < val < 0.7) else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    label = "Mean ranking score" if has_score else "Mean QED"
    ax.set_title(f"Scaffold family × pocket — {label}\n"
                 f"(families ordered by mean score, only non-singleton families shown)")
    fig.colorbar(im, ax=ax, shrink=0.6, label=label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_summary_dashboard(
    generated_df: pd.DataFrame, unique_df: pd.DataFrame, top_hits: pd.DataFrame,
    pocket_specs: list[PocketSpec], palette: dict[str, str], cfg: Config, out_path: Path,
) -> None:
    _ensure_plot_deps()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    if generated_df.empty:
        for ax in axes.flatten():
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    counts = generated_df.groupby("generator")["smiles"].nunique().sort_values(ascending=False)
    ax1.bar(counts.index, counts.values, color=[palette.get(k, "#4E79A7") for k in counts.index])
    ax1.set_title("Unique molecules per generator")
    ax1.tick_params(axis="x", rotation=35)
    scaffolds = (
        generated_df.loc[generated_df["scaffold"] != ""]
        .groupby("generator")["scaffold"].nunique()
        .reindex(counts.index).fillna(0)
    )
    ax2.bar(scaffolds.index, scaffolds.values, color=[palette.get(k, "#4E79A7") for k in scaffolds.index])
    ax2.set_title("Unique scaffolds per generator")
    ax2.tick_params(axis="x", rotation=35)
    for gen in counts.index:
        sub = generated_df[generated_df["generator"] == gen]
        ax3.scatter(sub["mw"], sub["qed"], s=28, alpha=0.7, color=palette.get(gen, "#4E79A7"), label=gen)
    ax3.set_title("Property landscape")
    ax3.set_xlabel("Molecular weight")
    ax3.set_ylabel("QED")
    ax3.legend(frameon=False)
    ax4.axis("off")
    text_lines = [
        f"Target: {cfg.target_name}",
        f"PDB: {cfg.pdb_id}",
        f"Pockets: {len(pocket_specs)}",
        f"Generated rows: {len(generated_df):,}",
        f"Unique molecules: {unique_df['smiles'].nunique():,}",
    ]
    if not top_hits.empty:
        best = top_hits.iloc[0]
        score_col = "rank_score" if pd.notna(best.get("rank_score")) else None
        text_lines += ["", "Top hit:", best["smiles"],
                       f"Generators: {best.get('generators', '')}"]
        if score_col:
            text_lines.append(f"Score: {best.get(score_col, 0):.3f}")
    ax4.text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_pocket_generator_heatmap(generated_df: pd.DataFrame, out_path: Path) -> None:
    _ensure_plot_deps()
    fig, ax = plt.subplots(figsize=(10, 6))
    if generated_df.empty:
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    pivot = generated_df.groupby(["pocket_id", "generator"])["qed"].max().unstack(fill_value=0)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Best QED per (pocket, generator)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_yield_matrix(generated_df: pd.DataFrame, out_path: Path) -> None:
    _ensure_plot_deps()
    fig, ax = plt.subplots(figsize=(10, 6))
    if generated_df.empty:
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    pivot = generated_df.groupby(["pocket_id", "generator"])["smiles"].nunique().unstack(fill_value=0)
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, str(pivot.values[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title("Unique molecules per (pocket, generator)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_scaffold_overlap(generated_df: pd.DataFrame, out_path: Path) -> None:
    _ensure_plot_deps()
    fig, ax = plt.subplots(figsize=(6, 5))
    generators = sorted(generated_df["generator"].unique()) if not generated_df.empty else []
    if not generators:
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    scaffold_sets = {
        gen: set(generated_df.loc[
            (generated_df["generator"] == gen) & (generated_df["scaffold"] != ""), "scaffold"
        ])
        for gen in generators
    }
    mat = np.zeros((len(generators), len(generators)), dtype=float)
    for i, g1 in enumerate(generators):
        for j, g2 in enumerate(generators):
            union = scaffold_sets[g1] | scaffold_sets[g2]
            inter = scaffold_sets[g1] & scaffold_sets[g2]
            mat[i, j] = len(inter) / len(union) if union else 0.0
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(generators, rotation=45, ha="right")
    ax.set_yticks(range(len(generators)))
    ax.set_yticklabels(generators)
    ax.set_title("Scaffold Jaccard overlap between generators")
    for i in range(len(generators)):
        for j in range(len(generators)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_pocket_druggability(
    unique_df: pd.DataFrame, pocket_specs: list[PocketSpec], out_path: Path,
) -> None:
    _ensure_plot_deps()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if not pocket_specs:
        ax1.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    pocket_ids = [s.pocket_id for s in pocket_specs]
    p2rank_scores = [s.p2rank_score for s in pocket_specs]
    best_scores = []
    for pid in pocket_ids:
        pocket_mols = unique_df[unique_df["pocket_ids"].str.contains(pid, na=False)]
        if pocket_mols.empty:
            best_scores.append(0.0)
        elif "rank_score" in pocket_mols and pocket_mols["rank_score"].notna().any():
            best_scores.append(float(pocket_mols["rank_score"].max()))
        else:
            best_scores.append(0.0)
    x = list(range(len(pocket_ids)))
    ax1.bar(x, best_scores, color="#4E79A7", alpha=0.8, label="Best mol score")
    ax1.set_ylabel("Best molecule score")
    ax1.set_title("Pocket druggability vs molecule quality")
    ax2 = ax1.twinx()
    ax2.plot(x, p2rank_scores, "ro-", label="P2Rank score")
    ax2.set_ylabel("P2Rank druggability")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pocket_ids, rotation=45, ha="right")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _get_devices() -> list[str]:
    if torch is not None and torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return devices if devices else ["cpu"]
    return ["cpu"]


def _build_palette(generator_names: list[str]) -> dict[str, str]:
    base = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948"]
    return {name: base[i % len(base)] for i, name in enumerate(sorted(generator_names))}


_WORKERS_PER_GPU: dict[str, int] = {
    "DiffSBDD":     1,
    "PocketXMol":   1,
    "PocketXMolAR": 1,
}


def _scored_candidates_path(run_dir: Path, scorer: str) -> Path:
    return (
        run_dir / "scored_candidates.csv"
        if scorer == "boltz"
        else run_dir / f"scored_candidates_{scorer}.csv"
    )


def _write_standard_analysis_csvs(run_dir: Path, generated_df: pd.DataFrame) -> None:
    try:
        build_scaffold_diversity_matrix(generated_df).to_csv(run_dir / "scaffold_diversity_matrix.csv")
        pocket_sensitivity(generated_df).to_csv(run_dir / "generator_pocket_sensitivity.csv", header=True)
        compute_budget_summary(generated_df).to_csv(run_dir / "compute_budget_summary.csv")
        scaffold_df = generated_df[generated_df["scaffold"] != ""]
        for pocket_id in scaffold_df["pocket_id"].unique():
            pocket_jaccard_matrix(
                scaffold_df[scaffold_df["pocket_id"] == pocket_id]
            ).to_csv(run_dir / f"scaffold_jaccard_{pocket_id}.csv")
    except Exception as exc:
        print(f"WARNING: analysis step failed: {exc}", flush=True)

    scaffold_df = generated_df.loc[generated_df["scaffold"] != ""]
    if not scaffold_df.empty:
        scaffold_df.groupby(["scaffold", "generator", "pocket_id"]).size().unstack(
            fill_value=0
        ).to_csv(run_dir / "scaffold_presence.csv")


def _write_presentation_analysis_csvs(run_dir: Path, merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    try:
        clustered = _cluster_molecule_fingerprints(
            merged,
            max_points=cfg.max_cluster_points,
            seed=cfg.seed,
        )
        _write_cluster_summary(clustered, run_dir / "cluster_summary.csv")
        _write_top_hit_enrichment(
            clustered, "generators", run_dir / "top_hit_enrichment_by_generator.csv", "generator"
        )
        _write_top_hit_enrichment(
            clustered, "pocket_ids", run_dir / "top_hit_enrichment_by_pocket.csv", "pocket"
        )
        _write_top_hit_enrichment(
            clustered, "chemical_cluster", run_dir / "top_hit_enrichment_by_cluster.csv", "cluster"
        )
        return clustered
    except Exception as exc:
        print(f"WARNING: presentation analysis failed: {exc}", flush=True)
        return merged.copy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Config, run_name: str | None = None, anchor_residue: str | None = None) -> Path:
    from chorus.analysis import (
        load_cached_run_tables,
        rebuild_analysis_outputs,
    )

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

    # Phase 0: Detect pockets
    pocket_specs = prepare_target_pockets(cfg.pdb_id, cfg, anchor_residue=anchor_residue)
    pocket_dir = run_dir / "pocket_specs"
    pocket_dir.mkdir(exist_ok=True)
    for spec in pocket_specs:
        write_pocket_spec_json(spec, pocket_dir / f"{spec.pocket_id}.json")

    # Phase 1: Generate
    all_results: dict[tuple[str, str], list[str]] = {}
    generator_errors: dict[str, str] = {}
    generator_output_root = run_dir / "generator_outputs"
    generator_output_root.mkdir(exist_ok=True)

    first_generators = build_generators(pocket_specs[0], cfg)
    generator_names = list(first_generators.keys())
    devices = _get_devices()
    n_gpus = len(devices)

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
        workers_per_gpu = _WORKERS_PER_GPU.get(gen_name, 1)
        max_workers = n_gpus * workers_per_gpu
        total_tasks = n_pockets

        print(
            f"\n=== {gen_name}: {n_pockets} pockets × {cfg.n_generate_per_model_per_pocket} samples "
            f"({n_gpus} GPUs × {workers_per_gpu} workers/GPU = {max_workers} parallel) ===",
            flush=True,
        )

        pocket_generators: list[tuple[PocketSpec, BasePocketGenerator]] = []
        for i, spec in enumerate(pocket_specs):
            generators = build_generators(spec, cfg)
            gen = generators[gen_name]
            gen.device = devices[i % n_gpus]
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

    # Build DataFrames
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

    # Phase 2: Score molecules
    all_smiles = unique_df["smiles"].tolist()
    scored_candidates_path = _scored_candidates_path(run_dir, scorer)
    if scored_candidates_path.exists():
        print(f"\n{scorer} scoring: resuming from {scored_candidates_path.name}", flush=True)
        merged = pd.read_csv(scored_candidates_path)
    else:
        if scorer == "boltz":
            oracle = BoltzCliOracle(pocket_specs[0], cfg)
            print(f"\nBoltz scoring {len(all_smiles)} unique molecules ...", flush=True)
            annotations = oracle.score(all_smiles)
            merged = unique_df.copy()
            merged["rank_score"] = [a.get("rank_score") for a in annotations]
            merged["affinity_probability_binary"] = [a.get("affinity_probability_binary") for a in annotations]
            merged["affinity_pred_value"] = [a.get("affinity_pred_value") for a in annotations]
            merged["scorer"] = "boltz"
        elif scorer == "rtmscore":
            print(f"\nRTMScore scoring poses for {len(all_smiles)} unique molecules ...", flush=True)
            merged = score_with_rtmscore(
                generated_df=generated_df,
                unique_df=unique_df,
                pocket_specs=pocket_specs,
                run_dir=run_dir,
                cfg=cfg,
            )
        elif scorer in ("none", "skip"):
            print("\nScoring skipped; ranking by QED only.", flush=True)
            merged = unique_df.copy()
            merged["rank_score"] = merged["qed"].fillna(0.0)
            merged["scorer"] = "none"
        else:
            raise ValueError(f"Unknown scorer: {cfg.scorer!r}. Expected boltz, rtmscore, or none.")
        merged.to_csv(scored_candidates_path, index=False)

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
