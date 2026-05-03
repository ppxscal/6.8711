# Chorus: Pocket-Conditioned Ligand Generation Panel

This project studies a simple question: when several strong pocket-conditioned ligand generators are given the same protein pockets, do they propose the same kinds of molecules, or do they reveal different pocket-specific chemical biases?

## One-Sentence Hypothesis

Pocket-conditioned generators are not interchangeable: pocket properties should influence which generator family produces the most promising chemotypes, and regions of agreement across generators should be more credible than generator-specific outliers.

## Named Contribution

We frame this project as a pocket-aware ensemble audit for structure-guided ligand design.

The contribution is not a new generator architecture. The contribution is a reproducible analysis workflow that:

1. compares multiple pocket-conditioned generators on the same detected pockets,
2. measures where they agree and disagree in chemical space,
3. links those differences back to pocket identity and pocket descriptors, and
4. surfaces chemically interpretable high-scoring groups for follow-up.

## Problem Framing

Most recent structure-based generative papers are organized around building a stronger single model. That is useful, but it leaves an important practical question underexplored: if different architectures produce different chemotypes for the same pocket, how should a medicinal chemist interpret that disagreement?

This project is built around that gap. We are asking whether generator disagreement is structured enough to learn something useful about:

- pocket-dependent generator preference,
- pocket-dependent chemotype preference,
- consensus versus model-specific chemical neighborhoods, and
- which generated regions look both high-scoring and plausibly synthesizable.

## Pipeline

The panel workflow is:

1. Select target proteins and names in `run_target_panel.sh`.
2. Detect candidate binding pockets with P2Rank.
3. Generate molecules independently for each detected pocket with DiffSBDD, DiffSBDDJoint, PocketXMol, and PocketXMolAR.
4. Deduplicate generated SMILES across generators and pockets.
5. Score generated poses with RTMScore.
6. Rebuild analysis CSVs and figures from cached generated/scored CSVs.

RTMScore is the scorer used by this workflow. The generators already produce 3D poses, so RTMScore keeps panel runs and repeated analysis rebuilds practical.

## Questions Each Analysis Answers

Every major figure or table should answer one question.

`figures/01_scores/score_distribution_summary.png`
Which generator and which pocket produce stronger candidates under the scoring function?

`figures/00_overview/aggregate_pca_overview.png`
Do generators populate different chemical regions, and where do high-scoring or synthesizable molecules concentrate?

`figures/03_families/family_structure_overview.png`
Can the generated molecules be grouped into interpretable chemical families, and do some families systematically score better?

`representative_cluster_smiles.csv`
What do the best-scoring and most representative molecules in each chemical group actually look like?

`source_pocket_predictability.csv` and `pocket_tanimoto_entropy.csv`
Does the conditioning pocket leave a detectable signature in the generated chemistry, or are some regions chemically shared across pockets?

Future router analysis:
Can simple pocket descriptors predict which generator should be prioritized for a new pocket?

## Running

From a fresh clone, build the local environments, model checkouts, and cached
weights:

```bash
bash setup.sh all
```

Confirm the expected environments and assets are present:

```bash
bash setup.sh check
```

Run a small pilot before launching a larger panel:

```bash
N_PER_POCKET=200 RUN_PREFIX=pilot_200 bash run_target_panel.sh
```

Run the current target panel with the script defaults. The public default is a
compact two-target panel, STK33 (`8VF6`) and GGCT (`2PN7`), at 300 samples per
generator-pocket:

```bash
bash run_target_panel.sh
```

By default, Chorus schedules one pocket task per visible GPU and caps parallel
work to the number of detected pockets. The run log prints the detected devices
and the subset used for each generator. If the launcher environment exposes more
GPUs than you want to use, restrict scheduling explicitly:

```bash
GPU_DEVICES=0,1,2 bash run_target_panel.sh
MAX_GPUS=3 bash run_target_panel.sh
```

Override targets explicitly with semicolon-separated `PDB_ID TARGET_NAME` pairs:

```bash
TARGETS_OVERRIDE="8VF6 STK33;2PN7 GGCT;4W9H VHL" bash run_target_panel.sh
```

Limit or select GPUs explicitly when running on a shared machine:

```bash
GPU_DEVICES=0,1,2 bash run_target_panel.sh
MAX_GPUS=3 bash run_target_panel.sh
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

## Current Run Status

As of May 2, 2026, the active run is the 300-sample-per-pocket panel for two
targets:

- `panel_20260502_stk33`: STK33, PDB `8VF6`.
- `panel_20260502_ggct`: GGCT, PDB `2PN7`.

These runs are still in progress, so the repository should not yet present the
300-sample outputs as final results. The earlier 200-sample setting is useful
as a pilot size for setup validation and smoke testing.

If a run is interrupted, rerun the same command with the same `RUN_PREFIX`.
Completed generator CSVs and scored CSVs are reused where possible. Use
`REFRESH=true` only when you intentionally want to discard and rebuild a run.

## Key Outputs

Each run writes to `results/<run_name>/`.

## Code Layout

The code is organized around the experiment flow:

- `config.py`: filesystem paths and run configuration.
- `pockets.py`: PDB download, P2Rank pocket detection, and cached pocket specs.
- `generators.py`: DiffSBDD conditional/joint, PocketXMol, and PocketXMolAR adapters.
- `scoring.py`: RTMScore pose scoring.
- `experiment.py`: top-level generation/scoring orchestration.
- `analysis.py`: cached rebuilds, analysis tables, and figures.

Protein structures are normalized before external tools consume them. mmCIF
downloads are converted to PDB when RCSB does not provide a direct PDB file,
and tools that only need protein atoms receive sanitized protein-only PDBs.
This avoids fixed-width PDB parsing failures from modern mmCIF ligand IDs while
keeping bound-ligand and pocket metadata available to Chorus.

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
- `figures/02_chemical_space/`: aggregate unique-molecule PCA colored by generator, conditioning pocket, RTMScore, and RA score.
- `figures/03_families/`: scaffold and ECFP family landscapes plus representative molecule/scaffold renderings.
- `figures/04_diagnostics/`: source-pocket and Tanimoto diagnostics for pocket-conditioning behavior.

Main figures:

- `figures/00_overview/main_story_overview.png`: compact stack for presentations and reports.
- `figures/00_overview/aggregate_pca_overview.png`: aggregate Morgan-fingerprint PCA colored by generator, conditioning pocket, RTMScore, and RA score.
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

`figures/01_scores/score_distribution_summary.png` answers: which generator and which conditioning pocket produce higher RTMScore candidates?

`figures/00_overview/aggregate_pca_overview.png` answers: do generators occupy different chemical regions, are high-scoring molecules concentrated in particular regions, and are those regions plausibly synthesizable according to RA score?

`figures/03_families/family_structure_overview.png` answers: can we group molecules into chemically related scaffold or ECFP modules, do some modules score better, and what do representative structures look like?

`figures/03_families/representative_cluster_molecules.png` answers: what do the highest-scoring chemical groups actually look like?

The source-pocket PCA is supplementary. It keeps duplicate source rows, so the same unique SMILES can appear once for each generator-pocket source that produced it. This is useful for testing whether the conditioning pocket shapes generated chemistry, but it is less clean than the aggregated unique-molecule PCA for the main narrative.

The pocket Tanimoto landscape is also supplementary. For each molecule, the pipeline compares its fingerprint to generated molecules from each source pocket. Low normalized entropy means the molecule is chemically closer to one pocket-specific neighborhood. High entropy means the molecule is similarly close to multiple pocket neighborhoods. The specificity margin is the best pocket-neighborhood similarity minus the second-best; high margin means a clearer pocket-specific assignment.

## Report Spine

If this becomes the report or presentation backbone, the recommended structure is:

1. Problem:
Do pocket-conditioned generators make meaningfully different proposals for the same protein pocket?

2. Hypothesis:
Pocket properties induce generator-specific chemotype preferences, so a multi-generator panel reveals information that a single-model benchmark hides.

3. Baseline:
Using one fixed generator for every pocket assumes the models are interchangeable.

4. Test:
Run the same pockets through DiffSBDD, DiffSBDDJoint, PocketXMol, and PocketXMolAR, score their outputs, and compare both score distributions and chemical-space occupancy.

5. Interpretation:
Measure whether disagreement is random noise or organized into pocket-linked chemical neighborhoods and scaffold families.

6. Optional ML extension:
Train a lightweight pocket-to-generator router only after establishing that there is a real pocket-specific ranking gap to predict.

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

Interpretation should be updated after the 300-sample two-target run finishes.
The intended analysis is comparative rather than absolute: we are measuring how
generator behavior differs under the same pocket detection and RTMScore scoring
setup.

The early pilot framing suggests three candidate findings to test:

1. PocketXMol and PocketXMolAR often occupy similar ligand regions, while DiffSBDD-family variants can explore distinct regions.
2. Score distributions vary by target and pocket; some pockets consistently yield stronger RTMScore candidates.
3. Generator advantage is target-dependent: PocketXMol-family models win in some regions/targets, while DiffSBDD-family variants can dominate others.

These are computational findings about generated and scored candidates, not
validated binders. The strongest report framing is therefore comparative and
diagnostic: this pipeline identifies where generators agree, where they are
biased, and which chemical regions deserve follow-up.

## Claims The Analysis Is Designed To Test

Once the current panel finishes, the analysis is designed to test whether:

- Different generators occupy different ligand-space regions for the same target panel.
- Generator performance is pocket-dependent under the chosen scoring setup.
- Some chemical groups are enriched for higher scores and are interpretable through representative structures.
- Agreement and disagreement across generators can be measured directly rather than inferred from aggregate benchmark numbers alone.

## Claims To Avoid

- We discovered true binders.
- We proved broad selectivity or solved promiscuity.
- A router trained on this panel generalizes broadly without stronger held-out validation.
- One model is universally best for pocket-conditioned ligand generation.
