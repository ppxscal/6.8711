# Chorus: Pocket-Conditioned Ligand Generation Panel

This project compares pocket-conditioned small-molecule generators on the same target proteins, then scores the generated poses with RTMScore and analyzes where high-scoring chemistry appears in fingerprint space.

## Hypothesis

Different pocket-conditioned generators explore different regions of ligand chemical space, and those generator-specific biases interact with the chosen protein pocket: some generator-pocket pairs should produce higher-scoring and more chemically coherent candidates than others.

## Pipeline

The panel workflow is:

1. Select target proteins and names in `run_target_panel.sh`.
2. Detect candidate binding pockets with P2Rank.
3. Generate molecules independently for each detected pocket with DiffSBDD, PocketXMol, and PocketXMolAR.
4. Deduplicate generated SMILES across generators and pockets.
5. Score generated poses with RTMScore.
6. Rebuild analysis CSVs and figures from cached generated/scored CSVs.

RTMScore is the practical default scorer because the generators already produce 3D poses, so scoring is much faster than Boltz-style protein-ligand prediction for every molecule. Boltz remains useful as a higher-cost oracle, but RTMScore enables larger panels and repeated analysis rebuilds.

## Running

Set up environments:

```bash
bash setup.sh all
```

Run the full target panel:

```bash
bash run_target_panel.sh
```

Regenerate analysis and figures from cached generated/scored CSVs:

```bash
bash rebuild_figures.sh
```

`rebuild_figures.sh` defaults to `JOBS=auto`, sizing parallel target rebuilds from CPU count, available memory, and the number of result folders. Override it when needed:

```bash
JOBS=1 bash rebuild_figures.sh
AUTO_MAX_JOBS=2 bash rebuild_figures.sh
```

Optional RA score setup:

```bash
bash setup.sh rascore
```

RAscore is installed into a separate `envs/uv/rascore` environment because its pretrained XGBoost model depends on older NumPy/pandas/scikit-learn/xgboost versions than the main analysis environment. Analysis uses that env automatically when it exists, caches scores to `ra_scores.csv`, and colors the aggregate PCA by RA score.

## Key Outputs

Each run writes to `results/<run_name>/`.

Core tables:

- `generated_by_generator.csv`: one row per generated molecule source, including generator and conditioning pocket.
- `unique_generated.csv`: deduplicated generated molecules.
- `scored_candidates_rtmscore.csv`: unique molecules with RTMScore pose scores.
- `top_unique_hits.csv`: highest-ranked unique molecules.
- `representative_cluster_smiles.csv`: sampled and representative SMILES from high-scoring chemical groups.

Figure folders:

- `figures/README.md`: per-run index of figure sections.
- `figures/00_overview/`: start here; compact presentation overview plus aggregate PCA overview.
- `figures/01_scores/`: RTMScore distributions, top hits, yield, and pocket/generator score summaries.
- `figures/02_chemical_space/`: aggregate unique-molecule PCA colored by generator, best-scoring pocket, RTMScore, and RA score.
- `figures/03_families/`: scaffold and ECFP family landscapes plus representative molecule/scaffold renderings.
- `figures/04_diagnostics/`: source-pocket and Tanimoto diagnostics for pocket-conditioning behavior.

Main figures:

- `figures/00_overview/main_story_overview.png`: compact stack for presentations and reports.
- `figures/00_overview/aggregate_pca_overview.png`: aggregate Morgan-fingerprint PCA colored by generator, best-scoring pocket, RTMScore, and RA score.
- `figures/03_families/family_structure_overview.png`: scaffold/ECFP family views supplemented with representative structures.
- `figures/03_families/representative_scaffold_families.png`: representative scaffold-family cores.
- `figures/03_families/representative_ecfp_groups.png`: representative ECFP-group molecules.

Supplementary diagnostics:

- `source_pocket_predictability.csv`: tests whether conditioning pocket can be predicted from fingerprints.
- `pocket_tanimoto_entropy.csv`: measures whether each molecule sits near one pocket-specific chemical neighborhood or several.
- `pocket_tanimoto_summary.csv`: pocket-level summary of Tanimoto neighborhood specificity.
- `scaffold_family_summary.csv`: score summaries for HDBSCAN scaffold families.
- `ecfp_group_summary.csv`: score summaries for hierarchical ECFP groups.

## How To Read The Analysis

Use the main story figure first.

`figures/01_scores/score_distribution_summary.png` answers: which generator and which best-scoring pocket produce higher RTMScore candidates?

`figures/00_overview/aggregate_pca_overview.png` answers: do generators occupy different chemical regions, are high-scoring molecules concentrated in particular regions, and are those regions plausibly synthesizable according to RA score?

`figures/03_families/family_structure_overview.png` answers: can we group molecules into chemically related scaffold or ECFP modules, do some modules score better, and what do representative structures look like?

`figures/03_families/representative_cluster_molecules.png` answers: what do the highest-scoring chemical groups actually look like?

The source-pocket PCA is supplementary. It keeps duplicate source rows, so the same unique SMILES can appear once for each generator-pocket source that produced it. This is useful for testing whether the conditioning pocket shapes generated chemistry, but it is less clean than the aggregated unique-molecule PCA for the main narrative.

The pocket Tanimoto landscape is also supplementary. For each molecule, the pipeline compares its fingerprint to generated molecules from each source pocket. Low normalized entropy means the molecule is chemically closer to one pocket-specific neighborhood. High entropy means the molecule is similarly close to multiple pocket neighborhoods. The specificity margin is the best pocket-neighborhood similarity minus the second-best; high margin means a clearer pocket-specific assignment.

## Representative Structures

`representative_cluster_smiles.csv` reports three grouping views:

- Morgan basins: HDBSCAN clusters over Morgan fingerprints after dimensionality reduction.
- ECFP groups: hierarchical groups built from Butina ECFP families.
- Scaffold families: HDBSCAN clusters over molecular scaffolds.

For each group, the table includes:

- `best_score_smiles`: highest RTMScore molecule in that group.
- `medoid_smiles`: most chemically central molecule by average Tanimoto similarity.
- `sampled_top_smiles`: several high-scoring examples.
- `top_scaffold`: most frequent Bemis-Murcko scaffold when one exists.

Use `best_score_smiles` for showing what scored well. Use `medoid_smiles` when you want a representative structure. Do not assume every group has a single clean scaffold; ECFP groups can capture related substituent patterns even when scaffold labels are messy.

## Current Interpretation

The early panel suggests three defensible findings:

1. PocketXMol and PocketXMolAR often occupy similar ligand regions, while DiffSBDD can explore distinct regions.
2. Score distributions vary by target and pocket; some pockets consistently yield stronger RTMScore candidates.
3. Generator advantage is target-dependent: PocketXMol-family models win in some regions/targets, while DiffSBDD can dominate others.

These are computational hypotheses, not validated binders. The strongest report framing is therefore comparative and diagnostic: this pipeline identifies where generators agree, where they are biased, and which chemical regions deserve follow-up.
