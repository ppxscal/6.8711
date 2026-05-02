"""Generator adapters and chemistry utilities for Chorus.

Each adapter wraps an upstream command-line generator and returns validated
SMILES while preserving the generator's saved 3D poses for downstream scoring.
"""
from __future__ import annotations

import random
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

import dataclasses
import hashlib
import numpy as np
import yaml
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

from chorus.pockets import protein_only_pdb
from chorus.runtime import env_python, run_command, visible_gpu_env

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Chemistry utilities
# ---------------------------------------------------------------------------

PAINS_PARAMS = FilterCatalogParams()
PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_FILTER = FilterCatalog(PAINS_PARAMS)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def is_valid_smiles(smiles: str) -> bool:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return False
    if PAINS_FILTER.HasMatch(mol):
        return False
    mw = Descriptors.MolWt(mol)
    return 120 <= mw <= 700


def get_scaffold(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return ""


def hash_smiles(smiles: str) -> str:
    return hashlib.sha256(smiles.encode()).hexdigest()[:16]


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def fp_array(smiles: str) -> np.ndarray | None:
    fp = morgan_fp(smiles)
    if fp is None:
        return None
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def ligand_properties(smiles: str) -> dict[str, float]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {}
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "qed": float(QED.qed(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rot_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "rings": float(Lipinski.RingCount(mol)),
    }


def smiles_from_sdf(path: Path) -> list[str]:
    smiles = []
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        try:
            clean = Chem.Mol(mol)
            Chem.SanitizeMol(clean)
            smi = Chem.MolToSmiles(Chem.RemoveHs(clean))
        except Exception:
            continue
        if smi:
            smiles.append(smi)
    return smiles


# ---------------------------------------------------------------------------
# Base generator
# ---------------------------------------------------------------------------

class BasePocketGenerator(ABC):
    def __init__(self, name: str, spec=None, device: str = "cpu"):
        self.name = name
        self.spec = spec
        self.device = device

    @abstractmethod
    def ready(self) -> bool: ...

    @abstractmethod
    def requirements(self) -> list[str]: ...

    @abstractmethod
    def generate(self, n: int, out_dir: Path | None = None) -> list[str]: ...

    def generate_validated(self, n: int, out_dir: Path | None = None) -> list[str]:
        raw = self.generate(n, out_dir=out_dir)
        seen: set[str] = set()
        valid = []
        for smi in raw:
            if smi in seen:
                continue
            seen.add(smi)
            if is_valid_smiles(smi):
                valid.append(smi)
        return valid[:n]


class MockPocketGenerator(BasePocketGenerator):
    def __init__(self, name: str, pool: list[str], seed: int):
        super().__init__(name=name, device="cpu")
        self.pool = pool
        self.rng = random.Random(seed)

    def ready(self) -> bool:
        return True

    def requirements(self) -> list[str]:
        return []

    def generate(self, n: int, out_dir: Path | None = None) -> list[str]:
        return [self.rng.choice(self.pool) for _ in range(n)]


MOCK_POOLS: dict[str, list[str]] = {
    "DiffSBDD": [
        "O=C(O)CCC(=O)Nc1ccccc1",
        "CCOC(=O)c1ccc(NC(=O)CCc2ccccc2)cc1",
        "O=C(NCc1ccccc1)C1CCNCC1",
        "COc1ccc2nc(NC(=O)CC3CC3)sc2c1",
        "CCN1CCN(C(=O)c2ccccc2F)CC1",
    ],
    "PocketXMol": [
        "O=C(Nc1ccccc1)c1ccc(O)cc1",
        "CC(=O)Nc1ccc(-c2ccccn2)cc1",
        "COc1ccc(NC(=O)c2cccs2)cc1",
        "O=C(NCc1ccco1)c1cccnc1",
        "CC(NC(=O)c1ccc(F)cc1)c1ccccc1",
    ],
    "PocketXMolAR": [
        "O=C(Nc1ccccc1)c1ccc(O)cc1",
        "CC(=O)Nc1ccc(-c2ccccn2)cc1",
        "COc1ccc(NC(=O)c2cccs2)cc1",
        "O=C(NCc1ccco1)c1cccnc1",
        "CC(NC(=O)c1ccc(F)cc1)c1ccccc1",
    ],
}


# ---------------------------------------------------------------------------
# DiffSBDD
# ---------------------------------------------------------------------------

def divisible_batch(n: int, max_batch: int) -> int:
    for bs in range(min(n, max_batch), 0, -1):
        if n % bs == 0:
            return bs
    return 1


class DiffSBDDCliGenerator(BasePocketGenerator):
    """Wrapper around DiffSBDD's generate_ligands.py CLI."""

    def __init__(self, spec, cfg, device: str = "cpu"):
        super().__init__(name="DiffSBDD", spec=spec, device=device)
        self._cfg = cfg
        self.repo = cfg.paths.models_dir / "diffsbdd"
        self.python = env_python("diffsbdd", cfg.paths)
        self.checkpoint = cfg.paths.checkpoints_dir / "diffsbdd" / "crossdocked_fullatom_cond.ckpt"

    def ready(self) -> bool:
        return self.repo.exists() and self.python.exists() and self.checkpoint.exists()

    def requirements(self) -> list[str]:
        return [str(p) for p in (self.repo, self.python, self.checkpoint) if not p.exists()]

    def generate(self, n: int, out_dir: Path | None = None) -> list[str]:
        out_dir = out_dir or (self._cfg.paths.results_dir / f"diffsbdd_{int(time.time())}")
        out_dir.mkdir(parents=True, exist_ok=True)
        outfile = out_dir / "diffsbdd_samples.sdf"
        if not self.spec.contact_residues:
            raise RuntimeError(f"DiffSBDD needs contact residues for {self.spec.pocket_id}")
        protein_pdb = protein_only_pdb(self.spec.full_pdb)
        env, _ = visible_gpu_env(self.device)
        run_command(
            [
                str(self.python), "generate_ligands.py", str(self.checkpoint),
                "--pdbfile", str(protein_pdb),
                "--outfile", str(outfile),
                "--resi_list", *self.spec.contact_residues,
                "--n_samples", str(n),
                "--batch_size", str(divisible_batch(n, 64)),
            ],
            cwd=self.repo,
            env=env,
            stream=True,
            quiet=self._cfg.quiet,
        )
        return smiles_from_sdf(outfile)


# ---------------------------------------------------------------------------
# PocketXMol
# ---------------------------------------------------------------------------

def pocketxmol_simple_noise_config() -> dict:
    return {
        "name": "sbdd",
        "num_steps": 100,
        "prior": "from_train",
        "level": {
            "name": "advance",
            "min": 0.0,
            "max": 1.0,
            "step2level": {
                "scale_start": 0.99999,
                "scale_end": 0.00001,
                "width": 3,
            },
        },
    }


def pocketxmol_ar_noise_config() -> dict:
    return {
        "name": "maskfill",
        "num_steps": 100,
        "ar_config": {
            "strategy": "refine",
            "r": 3,
            "threshold_node": 0.98,
            "threshold_pos": 0.91,
            "threshold_bond": 0.98,
            "max_ar_step": 10,
            "change_init_step": 1,
        },
        "prior": {
            "part1": "from_train",
            "part2": "from_train",
        },
        "level": {
            "part1": {
                "name": "uniform",
                "min": 0.6,
                "max": 1.0,
            },
            "part2": {
                "name": "advance",
                "min": 0.0,
                "max": 1.0,
                "step2level": {
                    "scale_start": 0.99999,
                    "scale_end": 0.00001,
                    "width": 3,
                },
            },
        },
    }


def write_pocketxmol_config(
    spec,
    n: int,
    cfg_dir: Path,
    seed: int,
    pocket_radius: float,
    sampling_mode: str,
) -> Path:
    cx, cy, cz = spec.center
    if sampling_mode == "simple":
        task = {"name": "sbdd", "transform": {"name": "sbdd"}}
        noise = pocketxmol_simple_noise_config()
    elif sampling_mode == "ar":
        task = {"name": "sbdd", "transform": {"name": "ar", "part1_pert": "small"}}
        noise = pocketxmol_ar_noise_config()
    else:
        raise ValueError(f"Unknown PocketXMol sampling mode: {sampling_mode}")

    config = {
        "sample": {
            "seed": seed,
            "batch_size": min(n, 500),
            "num_mols": n,
            "save_traj_prob": 0.0,
        },
        "data": {
            "protein_path": str(spec.pocket_pdb),
            "input_ligand": None,   # None triggers make_dummy_mol_with_coordinate via pocket_coord
            "is_pep": False,
            "pocket_args": {
                "pocket_coord": [float(cx), float(cy), float(cz)],
                "radius": pocket_radius,
            },
            "pocmol_args": {
                "data_id": f"sbdd_{spec.pocket_id}",
                "pdbid": spec.pocket_id,
            },
        },
        "transforms": {
            "featurizer_pocket": {
                "center": [float(cx), float(cy), float(cz)],
            },
            "variable_mol_size": {
                "name": "variable_mol_size",
                "num_atoms_distri": {
                    "strategy": "mol_atoms_based",
                    "mean": {"coef": 0, "bias": 28},
                    "std": {"coef": 0, "bias": 2},
                    "min": 5,
                },
            },
        },
        "task": task,
        "noise": noise,
    }
    cfg_path = cfg_dir / "pocketxmol_task.yml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return cfg_path


class PocketXMolCliGenerator(BasePocketGenerator):
    """Wrapper around PocketXMol's scripts/sample_use.py CLI."""

    generator_name = "PocketXMol"
    sampling_mode = "simple"

    def __init__(self, spec, cfg, device: str = "cpu"):
        super().__init__(name=self.generator_name, spec=spec, device=device)
        self._cfg = cfg
        self.repo = cfg.paths.models_dir / "pocketxmol"
        self.python = env_python("pocketxmol", cfg.paths)
        self.checkpoint = (
            cfg.paths.checkpoints_dir
            / "pocketxmol" / "data" / "trained_models" / "pxm" / "checkpoints" / "pocketxmol.ckpt"
        )

    def ready(self) -> bool:
        return self.repo.exists() and self.python.exists() and self.checkpoint.exists()

    def requirements(self) -> list[str]:
        return [str(p) for p in (self.repo, self.python, self.checkpoint) if not p.exists()]

    def generate(self, n: int, out_dir: Path | None = None) -> list[str]:
        out_dir = out_dir or (self._cfg.paths.results_dir / f"pocketxmol_{int(time.time())}")
        out_dir.mkdir(parents=True, exist_ok=True)

        chorus_cfg_dir = out_dir / "_config"
        chorus_cfg_dir.mkdir(parents=True, exist_ok=True)
        task_cfg = write_pocketxmol_config(
            self.spec,
            n,
            chorus_cfg_dir,
            self._cfg.seed,
            self._cfg.pocket_radius_angstrom,
            self.sampling_mode,
        )
        model_cfg = self.repo / "configs" / "sample" / "pxm.yml"

        env, local_device = visible_gpu_env(self.device)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.repo) if not existing_pp else f"{self.repo}:{existing_pp}"
        env.pop("TERM_PROGRAM", None)
        env.pop("TERM_PROGRAM_VERSION", None)

        run_command(
            [
                str(self.python), "scripts/sample_use.py",
                "--config_task", str(task_cfg),
                "--config_model", str(model_cfg),
                "--outdir", str(out_dir),
                "--device", local_device,
            ],
            cwd=self.repo,
            env=env,
            stream=True,
            quiet=self._cfg.quiet,
        )

        all_smiles: list[str] = []
        for csv_path in sorted(out_dir.rglob("gen_info.csv")):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                if "smiles" in df.columns:
                    all_smiles.extend(s for s in df["smiles"].dropna().tolist() if s)
            except Exception:
                pass
        if not all_smiles:
            for sdf in sorted(out_dir.rglob("*.sdf")):
                if sdf.name in ("reference_ligand.sdf", "input_mol.sdf"):
                    continue
                if "_inputs" in sdf.parts:
                    continue
                all_smiles.extend(smiles_from_sdf(sdf))
        return all_smiles


class PocketXMolARCliGenerator(PocketXMolCliGenerator):
    """PocketXMol SBDD with the upstream autoregressive/refine sampler."""

    generator_name = "PocketXMolAR"
    sampling_mode = "ar"


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

GENERATOR_REGISTRY: dict[str, type[BasePocketGenerator]] = {
    "DiffSBDD": DiffSBDDCliGenerator,
    "PocketXMol": PocketXMolCliGenerator,
    "PocketXMolAR": PocketXMolARCliGenerator,
}


def build_generators(spec, cfg) -> dict[str, BasePocketGenerator]:
    if cfg.generator_mode == "mock":
        return {
            name: MockPocketGenerator(name=name, pool=MOCK_POOLS[name], seed=cfg.seed + i)
            for i, name in enumerate(cfg.generators)
        }

    generators: dict[str, BasePocketGenerator] = {}
    missing: dict[str, list[str]] = {}
    for name in cfg.generators:
        cls = GENERATOR_REGISTRY.get(name)
        if cls is None:
            missing[name] = [f"Unknown generator: {name}"]
            continue
        gen = cls(spec, cfg)
        if gen.ready():
            generators[name] = gen
        else:
            missing[name] = gen.requirements()

    if missing:
        if not getattr(build_generators, "_warned", False):
            print("Skipping generators with missing components:")
            for name, reqs in missing.items():
                print(f"  skipping {name}: missing {reqs}")
            build_generators._warned = True  # type: ignore[attr-defined]

    if not generators:
        if cfg.allow_mock_fallback:
            print("No real generators ready. Falling back to mock.")
            import dataclasses
            return build_generators(spec, dataclasses.replace(cfg, generator_mode="mock"))
        missing_text = "\n".join(f"- {n}: {r}" for n, r in missing.items())
        raise RuntimeError(f"No generators are ready. Missing:\n{missing_text}")

    return generators
