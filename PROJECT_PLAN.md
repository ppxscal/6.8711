# Project Analysis and Presentation Plan

This document is the working spine for the class project. It is intentionally narrower than the full codebase. The goal is to keep the report and presentation centered on one question, a small number of motivated experiments, and claims that the current data can support.

## One-Sentence Hypothesis

Pocket-conditioned ligand generators are not interchangeable: for the same protein pockets, different generator architectures produce different chemotypes and different score versus synthesizability tradeoffs.

## Named Contribution

Pocket-Aware Ensemble Audit for Ligand Generation.

The contribution is not a new generative model. The contribution is a controlled analysis of how strong pocket-conditioned generators differ across the same targets and pockets, and what those differences imply for structure-guided small-molecule design.

## Problem Statement

Most recent structure-based generative papers focus on building a stronger single model and reporting aggregate benchmark metrics. That leaves an important practical question underexplored: if different architectures produce different chemistry for the same pocket, how should a user interpret that disagreement?

This project addresses that gap by running multiple strong generators on the same detected pockets, scoring their outputs, and analyzing:

- which generators perform better on which pockets,
- whether high-scoring regions are also synthetically plausible,
- which chemical families are shared versus model-specific,
- and whether generator disagreement is structured rather than random.

## Main Questions

Every final figure or table should answer one of these questions.

1. Are the generators interchangeable at the panel level?
2. Does generator preference depend on the target or pocket?
3. Do different generators occupy different regions of ligand chemical space?
4. Are the best-scoring regions also reasonably synthesizable?
5. What representative chemotypes explain the high-scoring families?

If a figure does not answer one of these, cut it.

## Final Analysis Plan

### 1. Panel Benchmark Summary

Goal:
Show that the problem is real before going deeper.

Question:
Do different generators produce different score and synthesizability profiles across the target panel?

Keep:

- score distribution summary by generator
- score distribution summary by best-scoring pocket
- generator-by-pocket heatmap or summary table
- RTMScore versus RA score comparison

Key output:
One compact figure or table that shows generator rankings are not constant across pockets and that score/synthesizability tradeoffs differ.

### 2. Focused Target Case Study

Goal:
Show what the disagreement looks like chemically.

Question:
When generators differ on a single target, do they occupy different chemical regions and support different chemotypes?

Keep:

- aggregate PCA overview
- family structure overview
- representative cluster or scaffold molecules

Preferred target:
Pick the cleanest and most visually interpretable target after the overnight run. Use one worked example, not several medium-strength examples.

### 3. Pocket-Specific Comparison

Goal:
Show that the pocket, not just the target, matters.

Question:
Within the same protein, do different pockets favor different generators or chemical families?

Keep:

- pocket-by-generator score summary
- best pocket coloring in aggregate PCA
- pocket-level family summaries if they are easy to read

This section should support the claim that generator behavior is pocket-dependent.

### 4. Synthesizability Analysis

Goal:
Make the project more practically relevant.

Question:
Do high-scoring molecules remain plausible from a retrosynthetic accessibility perspective?

Keep:

- RA score coloring on the aggregate PCA
- score versus RA summary by generator
- family-level score summary when it helps show a tradeoff

The main point is not to prove synthesizability, only to show whether some generators are winning by proposing chemistry that looks less practical.

### 5. Chemical Family Interpretation

Goal:
Move from abstract points to concrete structures.

Question:
What do the best-performing chemical groups actually look like?

Keep:

- representative scaffold families
- representative ECFP groups
- representative cluster SMILES table

Use these to show example molecules and family-level summaries, not as standalone decorative plots.

## Optional Work Only If Time Remains

### Router Analysis

Do this only if the overnight run gives enough clean pocket-level examples and the rest of the project is already stable.

Question:
Can simple pocket descriptors predict which generator should be prioritized?

Minimum defensible version:

- fixed-generator baseline
- oracle routing upper bound
- one simple model such as logistic regression or shallow random forest

Do not make this the core of the project unless the signal is clearly there.

## Plots To De-Emphasize or Cut

- plots that repeat the same story as the aggregate PCA
- diagnostics that are difficult to explain in one sentence
- anything that requires too much jargon for too little payoff
- any plot whose conclusion is only "there is some structure"

In particular, supplementary diagnostics should stay supplementary unless they directly strengthen one main claim.

## Claims We Can Defend

- Different generators can occupy different ligand-space regions on the same targets.
- Generator performance is target-dependent and pocket-dependent under the chosen scoring setup.
- High-scoring regions can differ in synthesizability profile.
- Representative chemical families can be extracted and compared across generators and pockets.

## Claims To Avoid

- We found real binders.
- We solved selectivity or promiscuity.
- One model is universally best.
- A lightweight router generalizes broadly unless held-out validation is strong.

## Limits Section

This should be stated directly in the report and presentation.

- RTMScore is a computational proxy, not experimental validation.
- P2Rank pockets are predicted pockets, not guaranteed binding sites.
- The project studies comparative behavior of generators under one scoring setup.
- Molecule counts can be large, but pocket-level generalization remains limited by the number of pockets and targets.

## Report Plan

Use the following section order.

1. Title
Pocket-Aware Ensemble Audit for Pocket-Conditioned Ligand Generation

2. Abstract
State the problem, the hypothesis, the compared generators, the main finding, and the practical takeaway in plain language.

3. Introduction
Explain that most work emphasizes stronger single generators, but users still need to know whether different architectures make different pocket-conditioned proposals and tradeoffs.

4. Related Work
Briefly cover CrossDocked-style structure-based benchmarks and the selected generators. Keep this short and focused on what is missing.

5. Methods
Explain the obvious approach and why it is insufficient:
run one generator and report aggregate metrics.

Then explain the fix:
run multiple generators on the same pockets, score them, and analyze score, synthesizability, and chemical-space structure.

6. Results
Use the same sequence as the Final Analysis Plan above.

7. Discussion
Interpret what the disagreements mean, what consensus may imply, and where the limits are.

8. Contributions
State explicitly what new knowledge this project adds.

9. Limitations and Future Work
Mention router analysis, more targets, stronger oracles, and experimental validation as future work, not current claims.

## Presentation Plan

Target length:
6 to 8 slides.

### Slide 1. Title and Hypothesis

Title:
Pocket-Aware Ensemble Audit for Ligand Generation

Say:
We asked whether strong pocket-conditioned generators are interchangeable, or whether different architectures produce different chemistry and practical tradeoffs on the same protein pockets.

### Slide 2. Why This Problem Matters

Question:
Why is a multi-generator analysis needed?

Show:
One simple schematic of the pipeline and a short statement that a single benchmark number hides disagreement between models.

### Slide 3. Benchmark Summary

Question:
Are the generators interchangeable at the panel level?

Show:
One summary figure or table with score and RA tradeoffs across generators.

Main message:
The problem is real because rankings and tradeoffs are not constant.

### Slide 4. Pocket Dependence

Question:
Does the preferred generator depend on the pocket?

Show:
Pocket-by-generator heatmap or compact pocket summary.

Main message:
Generator advantage changes across pockets, even within a target.

### Slide 5. Case Study Chemical Space

Question:
What does generator disagreement look like chemically?

Show:
Aggregate PCA overview for one target.

Main message:
Different generators occupy different regions, and score and RA can be overlaid on the same landscape.

### Slide 6. Representative Chemotypes

Question:
What chemistry explains the high-scoring families?

Show:
Representative scaffold or ECFP family structures.

Main message:
The differences are chemically interpretable, not just abstract score shifts.

### Slide 7. Main Takeaways

Say:

- pocket-conditioned generators are not interchangeable,
- generator choice changes score and synthesizability tradeoffs,
- and ensemble analysis exposes structure that single-model evaluation misses.

### Slide 8. Limits and Next Steps

Say:

- results depend on the scoring proxy,
- no experimental validation,
- router analysis is future work unless the overnight run makes it strong enough to include.

## Saturday to Monday Execution Plan

### Saturday

- clean the repository
- confirm the final target panel and generation settings
- launch the overnight run
- freeze the final figure set and file naming

### Sunday

- rebuild figures from cached outputs
- select the final benchmark figure, case-study figure, and representative chemotype figure
- draft the slides and report outline

### Monday

- tighten captions and wording
- remove weak or redundant figures
- rehearse with the limits stated clearly

## Decision Rule

If a new idea does not clearly improve one of the main questions above by Sunday, do not add it to the core project.
