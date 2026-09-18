# Figure attributions — Session 40

Every image used in the Session 40 deck is an original figure from a publicly available
paper or an official product page, reproduced here for teaching purposes with attribution.
No figure has been generated, redrawn, or altered other than by cropping to the figure
region. Each slide that uses one carries a visible credit line.

| File | Source | Figure | Licence |
|---|---|---|---|
| `fig_clip_contrastive_radford2021.png` | Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | Figure 1, panel (1) — contrastive pre-training | arXiv non-exclusive licence |
| `fig_rt1_architecture_brohan2022.png` | Brohan et al. (2022), *RT-1: Robotics Transformer for Real-World Control at Scale*, [arXiv:2212.06817](https://arxiv.org/abs/2212.06817) | Figure 1(a) — RT-1 architecture | arXiv non-exclusive licence |
| `fig_rt1_scale_examples_brohan2022.png` | Brohan et al. (2022), arXiv:2212.06817 | Figure 1(b) — dataset scale examples | arXiv non-exclusive licence |
| `fig_rt2_overview_zitkovich2023.png` | Zitkovich et al. (2023), *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) | Figure 1 — RT-2 overview | arXiv non-exclusive licence |
| `fig_openx_collage_oneill2023.png` | O'Neill et al. (2023), *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) | Figure 1 — embodiment collage | arXiv non-exclusive licence |
| `fig_openvla_overview_kim2024.png` | Kim, Pertsch, Karamcheti et al. (2024), *OpenVLA: An Open-Source Vision-Language-Action Model*, [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | Figure 1 — OpenVLA overview | arXiv non-exclusive licence |
| `fig_aloha_hardware_zhao2023.png` | Zhao, Kumar, Levine & Finn (2023), *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA), [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) | Figure 1 — ALOHA teleoperation hardware | arXiv non-exclusive licence |
| `fig_diffusionpolicy_representations_chi2023.png` | Chi et al. (2023/2024), *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | Figure 1 — explicit, implicit and diffusion policy representations | arXiv non-exclusive licence |
| `fig_pi0_overview_black2024.png` | Black et al., Physical Intelligence (2024), *pi_0: A Vision-Language-Action Flow Model for General Robot Control*, [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) | Figure 1 — pi-0 overview | arXiv non-exclusive licence |
| `fig_pi0_laundry_black2024.png` | Black et al., Physical Intelligence (2024), arXiv:2410.24164 | Figure 2 — laundry-folding rollout | arXiv non-exclusive licence |
| `fig_gr00t_datapyramid_nvidia2025.png` | NVIDIA (2025), *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) | Figure 1 — data pyramid | arXiv non-exclusive licence |
| `fig_cogact_realrobot_li2024.png` | Li et al. (2024), *CogACT: A Foundation Model for Vision-Language-Action Modeling*, [arXiv:2411.19650](https://arxiv.org/abs/2411.19650) | Figure 1(c) — real-robot evaluation | arXiv non-exclusive licence |
| `fig_saycan_overview_ahn2022.png` | Ahn et al. (2022), *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances* (SayCan), [arXiv:2204.01691](https://arxiv.org/abs/2204.01691) | Figure 1 — SayCan overview | arXiv non-exclusive licence |
| `fig_survey_structure_kawaharazuka2025.png` | Kawaharazuka, Oh, Yamada, Posner & Zhu (2025), *A Survey on Vision-Language-Action Models*, IEEE Access, [arXiv:2510.07077](https://arxiv.org/abs/2510.07077) | Figure 1 — survey structure | CC BY 4.0 (IEEE Access) |
| `fig_survey_timeline_kawaharazuka2025.png` | Kawaharazuka, Oh, Yamada, Posner & Zhu (2025), IEEE Access, arXiv:2510.07077 | Figure 2 — timeline of major VLA models | CC BY 4.0 (IEEE Access) |
| `fig_figure_helix_robots_2025.jpg` | Figure AI (2025), *Helix*, official product page, [figure.ai/news/helix](https://www.figure.ai/news/helix) | Product photo — two Figure robots collaborating | Official press/marketing image |

## Note on the box-and-arrow and table diagrams in the deck

Slides that show pipelines, loops, tables, or timelines as rounded rectangles, arrows, and
native tables (for example the closed-loop diagram, the compounding-error drift diagram,
the training-stage pipeline, the dataset and model-summary tables, the traditional-pipeline
comparison, and the challenges list) are **native PowerPoint shapes**, not images, built with
the same house-style helper module used across the Dhruv Sessions decks. No raster
illustration was generated for any of these.

## How the figures were obtained

Each source PDF was downloaded from arXiv and the figure region was cropped at 300 DPI with
PyMuPDF (`page.get_pixmap(dpi=300, clip=fitz.Rect(...))`), iterating the crop box against a
live preview render until the source paper's own (often cut-off) caption text was fully
excluded, since house style requires the credit line to be a separate slide caption textbox,
never baked into the image. The Figure AI Helix photo was downloaded directly from the
official product page rather than cropped from a PDF.
