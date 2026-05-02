"""Protein download, pocket detection, and pocket-spec caching."""
from __future__ import annotations

import csv
import json
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chorus.config import Config
from chorus.runtime import run_command

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


def pdb_residue_fields(line: str) -> tuple[str, str, str]:
    """Return chain, residue name, and residue number from fixed-width or shifted PDB lines."""
    chain = line[21].strip() or "A"
    resname = line[17:20].strip()
    resseq = line[22:26].strip()
    if line.startswith("HETATM") and not resseq.lstrip("-").isdigit():
        parts = line.split()
        if len(parts) >= 6:
            return parts[4], parts[3], parts[5]
    return chain, resname, resseq


def download_pdb(pdb_id: str, cfg: Config) -> Path:
    out_path = cfg.paths.data_dir / f"{pdb_id.lower()}.pdb"
    if out_path.exists():
        return out_path
    cif_path = out_path.with_suffix(".cif")
    if cif_path.exists():
        return convert_cif_to_pdb(cif_path, out_path, pdb_id)
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
    print(f"  .pdb not available, downloading mmCIF from {cif_url}")
    urllib.request.urlretrieve(cif_url, cif_path)
    return convert_cif_to_pdb(cif_path, out_path, pdb_id)


def convert_cif_to_pdb(cif_path: Path, out_path: Path, pdb_id: str) -> Path:
    """Convert an RCSB mmCIF download to the PDB format expected downstream."""
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
            chain, resname, resseq = pdb_residue_fields(line)
            groups.setdefault((chain, resname, resseq), []).append(line.rstrip("\n"))
    return groups


def find_ligand_candidates(pdb_path: Path) -> list[LigandCandidate]:
    groups: dict[tuple[str, str, str], list[str]] = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            chain, resname, resseq = pdb_residue_fields(line)
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
            chain, resname, resseq = pdb_residue_fields(line)
            if chain != chain_id:
                continue
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
            chain, _, resseq = pdb_residue_fields(line)
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


def download_file_atomic(url: str, dest: Path) -> None:
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def validate_p2rank_archive(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        tar.getmembers()


def ensure_p2rank(cfg: Config) -> Path:
    prank_bin = cfg.paths.p2rank_dir / f"p2rank_{P2RANK_VERSION}" / "prank"
    if prank_bin.exists():
        return prank_bin
    cfg.paths.p2rank_dir.mkdir(parents=True, exist_ok=True)
    install_dir = prank_bin.parent
    archive = cfg.paths.p2rank_dir / f"p2rank_{P2RANK_VERSION}.tar.gz"
    needs_download = not archive.exists()
    if not needs_download:
        try:
            validate_p2rank_archive(archive)
        except (tarfile.TarError, EOFError, OSError) as exc:
            print(f"P2Rank archive is invalid ({exc}); re-downloading", flush=True)
            archive.unlink(missing_ok=True)
            shutil.rmtree(install_dir, ignore_errors=True)
            needs_download = True
    if needs_download:
        print(f"Downloading P2Rank {P2RANK_VERSION}")
        download_file_atomic(P2RANK_URL, archive)
        try:
            validate_p2rank_archive(archive)
        except (tarfile.TarError, EOFError, OSError):
            archive.unlink(missing_ok=True)
            raise
    shutil.rmtree(install_dir, ignore_errors=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(cfg.paths.p2rank_dir)
    except Exception:
        shutil.rmtree(install_dir, ignore_errors=True)
        raise
    if not prank_bin.exists():
        raise RuntimeError(f"P2Rank extraction failed: expected {prank_bin}")
    return prank_bin


def protein_only_pdb(pdb_path: Path) -> Path:
    """Write a P2Rank-safe PDB containing only protein atom records."""
    pdb_path = pdb_path.resolve()
    out_path = pdb_path.with_name(f"{pdb_path.stem}_protein_only.pdb")
    if out_path.exists() and out_path.stat().st_mtime >= pdb_path.stat().st_mtime:
        return out_path

    atom_lines = []
    with open(pdb_path) as handle:
        for line in handle:
            if line.startswith("ATOM"):
                atom_lines.append(line.rstrip("\n"))
    if not atom_lines:
        raise RuntimeError(f"No protein ATOM records found in {pdb_path}")

    out_path.write_text("\n".join(atom_lines) + "\nEND\n")
    return out_path


def run_p2rank(pdb_path: Path, cfg: Config) -> list[dict]:
    prank_bin = ensure_p2rank(cfg)
    out_dir = pdb_path.parent / f"{pdb_path.stem}_p2rank"
    out_dir.mkdir(parents=True, exist_ok=True)
    p2rank_pdb = protein_only_pdb(pdb_path)
    run_command([str(prank_bin), "predict", "-f", str(p2rank_pdb), "-o", str(out_dir)],
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


def parse_p2rank_residues(residue_str: str) -> list[str]:
    parts = []
    for token in residue_str.split():
        if "_" in token:
            chain, resseq = token.rsplit("_", 1)
            parts.append(f"{chain}:{resseq}")
    return parts


def ligand_from_pocket_residues(
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
    first_residues = parse_p2rank_residues(filtered[0]["residue_ids"])
    primary_chain = first_residues[0].split(":")[0] if first_residues else "A"
    seq_cache: dict[str, str] = {}
    specs = []
    for i, pocket in enumerate(filtered):
        pocket_id = f"pocket_{i + 1}"
        center: tuple[float, float, float] = (pocket["center_x"], pocket["center_y"], pocket["center_z"])
        p2rank_residues = parse_p2rank_residues(pocket["residue_ids"])
        pocket_chain = p2rank_residues[0].split(":")[0] if p2rank_residues else primary_chain
        if pocket_chain not in seq_cache:
            try:
                seq_cache[pocket_chain] = extract_chain_sequence(pdb_path, pocket_chain)
            except RuntimeError:
                seq_cache[pocket_chain] = seq_cache.get(primary_chain, "")
        ligand = ligand_from_pocket_residues(pdb_path, center, p2rank_residues, pocket["name"])
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
        return prepare_single_pocket(pdb_id, full_pdb, cfg, anchor_residue)
    candidates = find_ligand_candidates(full_pdb)
    if candidates and cfg.ligand_resname_preference:
        return prepare_single_pocket(pdb_id, full_pdb, cfg, anchor_residue)
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
        return prepare_single_pocket(pdb_id, full_pdb, cfg, anchor_residue)


def prepare_single_pocket(
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
