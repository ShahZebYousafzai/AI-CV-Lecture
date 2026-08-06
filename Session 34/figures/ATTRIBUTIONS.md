# Figure attributions — Session 34

Every image used in the Session 34 decks is an original figure from a publicly available
paper, reproduced here for teaching purposes with attribution. No figure has been generated,
redrawn, or altered other than by cropping to the figure region. Each slide that uses one
carries a visible credit line.

| File | Source | Figure | Licence |
|---|---|---|---|
| `fig_hyde_gao2022.png` | Gao, Ma, Lin & Callan (2022), *Precise Zero-Shot Dense Retrieval without Relevance Labels*, [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) — ACL 2023 | Figure 1 — illustration of the HyDE model | arXiv non-exclusive licence |
| `fig_ragfusion_rackauckas2024.png` | Rackauckas (2024), *RAG-Fusion: A New Take on Retrieval-Augmented Generation*, [arXiv:2402.03367](https://arxiv.org/abs/2402.03367) | Figure 1 — high-level RAG-Fusion process | arXiv non-exclusive licence |
| `fig_pipeline_stages_cascade2026.png` | Singh, Allam Reddy & Chopra (2026), *Beyond the Reranker: Do RAG Retrieval Enhancements Help Once a Strong Reranker Is Present?*, [arXiv:2606.28367](https://arxiv.org/abs/2606.28367) | Figure 1 — methods by pipeline stage | CC BY-NC-ND 4.0 |
| `fig_clip_contrastive_radford2021.png` | Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | Figure 1, panel (1) — contrastive pre-training | arXiv non-exclusive licence |
| `fig_modality_gap_liang2022.png` | Liang et al. (2022), *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*, [arXiv:2203.02053](https://arxiv.org/abs/2203.02053) — NeurIPS 2022 | Figure 1(b), CLIP panel — UMAP of paired image and text embeddings | arXiv non-exclusive licence |
| `fig_colpali_faysse2024.png` | Faysse et al. (2024), *ColPali: Efficient Document Retrieval with Vision Language Models*, [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) — ICLR 2025 | Figure 1 — standard retrieval vs ColPali | arXiv non-exclusive licence |

## Note on the box-and-arrow diagrams in the decks

Slides that show pipelines as rounded rectangles and arrows (e.g. the parent-document
retriever, the hybrid fusion diagram, the multimodal indexing flow) are **native PowerPoint
shapes**, not images. They follow the same convention used in the Session 32a and 32b decks
and are fully editable in PowerPoint. No raster illustration was generated for this session.

## How the figures were obtained

Each source PDF was downloaded from arXiv and the figure region was cropped at 300–450 DPI
with PyMuPDF (`page.get_pixmap(dpi=..., clip=fitz.Rect(...))`). Crop rectangles, in PDF
points, on the page listed:

| File | Page (0-indexed) | Clip rect (x0, y0, x1, y1) | DPI |
|---|---|---|---|
| `fig_hyde_gao2022.png` | 1 | 58, 58, 538, 198 | 300 |
| `fig_ragfusion_rackauckas2024.png` | 2 | 189, 80, 429, 298 | 330 |
| `fig_pipeline_stages_cascade2026.png` | 1 | 55, 60, 558, 184 | 300 |
| `fig_clip_contrastive_radford2021.png` | 1 | 50, 68, 302, 251 | 300 |
| `fig_modality_gap_liang2022.png` | 1 | 166, 116, 248, 204 | 450 |
| `fig_colpali_faysse2024.png` | 1 | 50, 60, 562, 318 | 300 |
