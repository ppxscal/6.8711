"""Configuration objects and filesystem layout for Chorus experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path("/data2/ppxscal")


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
    p2rank_dir:      Path

    @classmethod
    def from_root(cls, root: Path | str) -> "Paths":
        root = Path(root).resolve()
        cache = root / "cache"
        return cls(
            root=root,
            data_dir=root / "data",
            results_dir=root / "results",
            models_dir=root / "models",
            checkpoints_dir=root / "checkpoints",
            tools_dir=root / "tools",
            envs_dir=root / "envs",
            cache_dir=cache,
            p2rank_dir=root / "tools" / "p2rank",
        )

    def makedirs(self) -> None:
        for path in (
            self.data_dir, self.results_dir, self.models_dir,
            self.checkpoints_dir, self.tools_dir, self.envs_dir,
            self.cache_dir,
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

    n_generate_per_model_per_pocket: int = 1250

    pocket_radius_angstrom:    float      = 10.0
    pocket_bbox_size_angstrom: float      = 23.0
    p2rank_min_score:          float      = 0.5
    p2rank_max_pockets:        int        = 4
    ligand_resname_preference: str | None = None


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

    paths: Paths = field(default_factory=lambda: Paths.from_root(DEFAULT_ROOT))

