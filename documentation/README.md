# Documentation

This folder contains persistent project documentation and extracted notebook outputs for the speech emotion direction project.

## Files

- `research_report.md`: paper-style report draft with figures, tables, methods, results, limitations, and conclusion.
- `assets/figures/`: extracted PNG figures from executed notebooks `01` through `16`.
- `assets/notebook_text_outputs/`: cleaned text outputs from executed notebooks, useful for auditing exact printed metrics.
- `assets/tables/key_results.csv`: compact table of headline metrics used in the report.
- `assets/tables/notebook_execution_status.csv`: execution status and extracted-figure count for each notebook.
- `assets/figure_manifest.csv`: mapping from every extracted figure back to its source notebook and cell.

## Notes

The original Colab/Drive artifacts are still useful, but this folder makes the key report evidence persist directly inside the git repository. Large model checkpoints, raw audio, and local LaTeX/Overleaf source exports are intentionally not copied here. The central report narrative now uses notebooks `14`, `15`, and `16` as the strongest causal evidence: mid-transformer direction intervention, same-context activation patching, and robustness/negative-control validation.
