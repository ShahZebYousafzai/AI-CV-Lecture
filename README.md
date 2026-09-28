# AI & Computer Vision — Training Curriculum (as delivered)

A structured one-on-one training program covering Computer Vision, Deep Learning, NLP, LLMs, Generative AI, and AI Agents. Sessions run **1–1.5 hours** with a mixed theory / hands-on split.

> **This README reflects what was actually delivered** (Sessions 1–40, matching the `Session 1` – `Session 40` folders in this repo), not the original forward-looking curriculum plan. Delivery diverged substantially from that original plan starting around Session 7 — see [“Departures from the original plan”](#departures-from-the-original-plan) below.

---

## Program Overview

| Stat | Value |
|---|---|
| Phases | 10 |
| Sessions delivered | 40 |
| Hours logged (est.) | ~40–60 |
| Key projects with saved deliverables | 4 |
| Sessions 41+ | Not yet scheduled |

**Tools & Stack** (confirmed in delivered materials)

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| CV Libraries | OpenCV, Ultralytics YOLOv8, MediaPipe, PaddleOCR/EasyOCR |
| DL Framework | PyTorch + torchvision |
| GenAI Stack | HuggingFace Diffusers, Transformers, PEFT (QLoRA) |
| LLM / Agent Stack | LangChain, LangGraph, FAISS, Streamlit |
| Deployment | FastAPI, Docker, AWS (ECR/EC2) — see Session 39 |

---

## Phases at a Glance

| Phase | Title | Sessions | Count |
|---|---|---|---|
| Phase 1 | Foundations | 1–6 | 6 |
| Phase 2 | Deep Learning Essentials | 7–9 | 3 |
| Phase 3 | Core CV Tasks | 10–13 | 4 |
| Phase 4 | Advanced Vision & Transformers | 14–18 | 5 |
| Phase 5 | NLP Foundations | 19–24 | 6 |
| Phase 6 | Generative AI & Foundation Models | 25–28 | 4 |
| Phase 7 | LangChain & Chatbots | 29–31 | 3 |
| Phase 8 | RAG Systems | 32–34 | 3 |
| Phase 9 | AI Agents | 35–37 | 3 |
| Phase 10 | Production, Deployment & Emerging Topics | 38–40 | 3 |

---

## Session Log

### Phase 1 — Foundations (Sessions 1–6)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 1 | Course Intro + Python & NumPy Refresh | Mixed | [Slides](Session%201/Session_01_Python_Environment_Setup.pptx) · [Notebook](Session%201/Session_01_NumPy_Practical.ipynb) | First session |
| 2 | Image Fundamentals | Theory | [Slides](Session%202/Session_02_Image_Fundamentals.pptx) |  |
| 3 | OpenCV Basics I | Practical | [Notebook](Session%203/Session_03_OpenCV_Basics.ipynb) |  |
| 4 | OpenCV Basics II | Practical | [Notebook](Session%204/Session_04_OpenCV_Basics_2.ipynb) |  |
| 5 | Filtering & Edge Detection | Mixed | [Slides](Session%205/Session_05_Filtering_and_Edge_Detection.pptx) · [Notebook](Session%205/Session_05_Filtering_and_Edge_Detection.ipynb) |  |
| 6 | Geometric Transforms | Practical | [Notebook](Session%206/Session_06_Geometric_Transforms.ipynb) |  |

### Phase 2 — Deep Learning Essentials (Sessions 7–9)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 7 | PyTorch Fundamentals — MLP on MNIST | Mixed | [Slides](Session%207/Session_07.pptx) · [Notebook](Session%207/Session_07_PyTorch_MLP_MNIST.ipynb) | Replaces the originally planned “Contours & Shape Analysis” session |
| 8 | CNNs — CIFAR-10 | Mixed | [Slides](Session%208/Session_8_Slides.pptx) · [Notebook](Session%208/Session_8__CNN_CIFAR10.ipynb) | Replaces the originally planned “Document Scanner” milestone |
| 9 | Transfer Learning — Cats vs Dogs | Mixed | [Slides](Session%209/Session_9_Slides.pptx) · [Notebook](Session%209/Session_9_Transfer_Learning_CatsDogs.ipynb) | Trained checkpoints + TorchScript export saved |

### Phase 3 — Core CV Tasks (Sessions 10–13)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 10 | Object Detection — YOLOv8 | Mixed | [Slides](Session%2010/Session_10_Object_Detection_YOLO.pptx) · [Notebook](Session%2010/Session_10_Object_Detection_YOLO.ipynb) |  |
| 11 | Segmentation | Mixed | [Slides](Session%2011/Session_11_Segmentation.pptx) · [Notebook](Session%2011/Session_11_Segmentation.ipynb) |  |
| 12 | Pose Estimation | Mixed | [Slides](Session%2012/Session_12_Pose_Estimation.pptx) · [Notebook](Session%2012/Session_12_Pose_Estimation.ipynb) |  |
| 13 | OCR & Face Recognition | Practical | [OCR Notebook](Session%2013/Session_13A_OCR_and_Text_Detection.ipynb) · [Face Recognition Notebook](Session%2013/Session_13B_Face_Recognition_Pipeline.ipynb) | Two-part session, no slides |

### Phase 4 — Advanced Vision & Transformers (Sessions 14–18)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 14 | Transformers & Attention (Vision) | Theory | [Slides](Session%2014/Session14_Transformers_Attention.pptx) | Slides only, no notebook |
| 15 | Video Analytics I | Theory | [Slides](Session%2015/Session_15_Video_Analytics_I.pptx) | Slides only, no notebook |
| 16 | Vision Transformers Lab | Practical | [Notebook](Session%2016/Session_16_ViT_Lab.ipynb) | Notebook only, no slides |
| 17 | Depth Estimation & 3D | Mixed | [Slides](Session%2017/Session17_DepthEstimation_3D.pptx) · [Script](Session%2017/MiDaS%20Depth/main.py) |  |
| 18 | Multi-Object Tracking | Practical | [Notebook](Session%2018/Session_18_MultiObject_Tracking.ipynb) | Notebook only, no slides |

### Phase 5 — NLP Foundations (Sessions 19–24)

> **Not part of the original curriculum plan — added to build an NLP foundation from first principles ahead of the LLM/GenAI phases.**

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 19 | NLP & Text Processing Fundamentals | Mixed | [Slides](Session%2019/Session_19_NLP_Text_Processing_Fundamentals.pptx) · [Notebook](Session%2019/Session_19_NLP_Text_Processing_Practical.ipynb) |  |
| 20 | Text Representation Fundamentals | Mixed | [Slides](Session%2020/Session_20_Text_Representation_Fundamentals.pptx) · [Notebook](Session%2020/Session_20_Text_Representation_Practical.ipynb) |  |
| 21 | Word Embeddings | Mixed | [Slides](Session%2021/Session_21_Word_Embeddings.pptx) · [Notebook](Session%2021/Session_21_Word_Embeddings_Practical.ipynb) |  |
| 22 | Neural Networks for NLP | Mixed | [Slides](Session%2022/Session_22_Neural_Networks_for_NLP.pptx) · [Notebook](Session%2022/Session_22_Neural_Networks_for_NLP_Practical.ipynb) |  |
| 23 | Attention and Transformers (NLP) | Mixed | [Slides](Session%2023/Session_23_Attention_and_Transformers.pptx) · [Notebook](Session%2023/Session_23_Attention_and_Transformers_Practical.ipynb) |  |
| 24 | Language Models | Theory | [Slides](Session%2024/Session_24_Language_Models.pptx) | Slides only, no notebook |

### Phase 6 — Generative AI & Foundation Models (Sessions 25–28)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 25 | Diffusion Models & Stable Diffusion | Mixed | [Slides](Session%2025/Session_25_Diffusion_Models_and_Stable_Diffusion.pptx) · [Notebook](Session%2025/Session_25_Diffusion_Models_and_Stable_Diffusion_Practical.ipynb) |  |
| 26 | CLIP | Mixed | [Slides](Session%2026/Session_26_CLIP.pptx) · [Notebook](Session%2026/Session_26_CLIP_Practical.ipynb) |  |
| 27 | SAM — Segment Anything | Mixed | [Slides](Session%2027/Session_27_SAM.pptx) · [Notebook](Session%2027/Session_27_SAM3_Practical.ipynb) | Practical notebook uses SAM3 |
| 28 | Pretraining, Fine-tuning & QLoRA | Mixed | [Slides](Session%2028/Session_28_Pretraining_FineTuning_and_Efficient_FineTuning.pptx) · [Notebook](Session%2028/Session_28_QLoRA_Finetuning_Practical.ipynb) |  |

### Phase 7 — LangChain & Chatbots (Sessions 29–31)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 29 | LangChain Fundamentals | Mixed | [Slides](Session%2029/Session29_LangChain.pptx) · [Notebook](Session%2029/Session29_LangChain_Notebook.ipynb) |  |
| 30 | Prompt Engineering & Chatbot Architecture | Mixed | [Slides A](Session%2030/Session30a_PromptEngineering.pptx) · [Notebook](Session%2030/Session30a_PromptEngineering_Notebook.ipynb) · [Slides B](Session%2030/Session30b_ChatbotArchitecture.pptx) |  |
| 31 | Chatbot Labs — LangChain/Streamlit + Tools & Guardrails | Practical | [Lab 1 Guide](Session%2031/Session31_Lab1_Chatbot_LangChain_Streamlit_Guide.md) · [Lab 2 Guide](Session%2031/Session31_Lab2_Tools_StructuredOutput_Guardrails_Guide.md) · [Starter project](Session%2031/starter) | Full starter project (agent, tools, guardrails, config) — no slides |

### Phase 8 — RAG Systems (Sessions 32–34)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 32 | RAG Fundamentals & Vector Databases | Mixed | [Slides A](Session%2032/Session32a_RAG_Fundamentals.pptx) · [Slides B](Session%2032/Session32b_VectorDatabases.pptx) · [Notebook](Session%2032/Session32b_VectorDatabases_Notebook.ipynb) | FAISS index + doc store saved (handbook.index, handbook_docs.pkl) |
| 33 | RAG Pipeline Lab | Practical | [Notebook](Session%2033/Session33_RAG_Pipeline_Lab.ipynb) · [Guide](Session%2033/Session33_RAG_Pipeline_Lab_Guide.md) | Notebook + written guide, no slides |
| 34 | Advanced & Multimodal RAG | Mixed | [Slides A](Session%2034/Session34a_Advanced_RAG.pptx) · [Slides B](Session%2034/Session34b_Multimodal_RAG.pptx) · [Notebook](Session%2034/Session34_Multimodal_RAG_Practical.ipynb) | Corpus: Attention, ViT and LoRA papers with extracted figures |

### Phase 9 — AI Agents (Sessions 35–37)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 35 | AI Agents & LangGraph | Theory | [Slides](Session%2035/Session35_AI_Agents_LangGraph.pptx) | Slides only, no notebook |
| 36 | Multi-Agent Systems | Theory | [Slides](Session%2036/Session36_Multi_Agent_Systems.pptx) | Slides only, no notebook |
| 37 | LangGraph Lab II & Agentic RAG | Practical | [LangGraph Notebook](Session%2037/Session37a_LangGraph_II.ipynb) · [Agentic RAG Notebook](Session%2037/Session37b_Agentic_RAG.ipynb) | Notebooks only, no slides |

### Phase 10 — Production, Deployment & Emerging Topics (Sessions 38–40)

| Session | Topic | Type | Materials | Notes |
|---|---|---|---|---|
| 38 | Model Optimisation | Theory | [Slides](Session%2038/Session38_Model_Optimization.pptx) · [Research notes](Session%2038/Session38_Research_and_Sources.md) · [Delivery transcript](Session%2038/Session38_Delivery_Transcript.docx) | Quantisation, pruning, precision benchmarking (DeepCompression-style pipeline) |
| 39 | FastAPI Serving & Cloud Deployment | Mixed | [Slides](Session%2039/Session39_FastAPI_Serving_and_Cloud_Deployment.pptx) · [yolo-serve project](Session%2039/yolo-serve) | Most complete delivered project — runs locally, in Docker, and on AWS EC2 unchanged (20 tests) |
| 40 | Vision-Language-Action Models | Theory | [Slides](Session%2040/Session40_Vision_Language_Action_Models.pptx) · [Delivery transcript](Session%2040/Session40_Delivery_Transcript.docx) | RT-1/RT-2, OpenVLA, π0, GR00T, SayCan, ALOHA, Diffusion Policy — not part of the original plan |

---

## Departures from the Original Plan

- **Sessions 7–8** replaced the originally planned “Contours & Shape Analysis” and “Document Scanner” milestone with a PyTorch MLP (MNIST) session and a CNN (CIFAR-10) session.
- **Phase 5 (Sessions 19–24)** is a 6-session NLP-foundations track (text processing → representations → embeddings → NLP attention → language models) that isn't in the original 83-session plan at all.
- **Session 40** covers Vision-Language-Action / robotics models (RT-1/RT-2, OpenVLA, π0, GR00T, SayCan, ALOHA, Diffusion Policy), also not in the original plan.
- No formal milestone/capstone checkpoints appear in the delivered materials, though four phases do have a real saved deliverable: the Cats-vs-Dogs classifier (Phase 2), the tool-calling support chatbot (Phase 7), the FAISS-backed RAG system (Phase 8), and the yolo-serve service deployed to AWS (Phase 10).
- **Sessions 41+** are not yet scheduled — the original forward plan is no longer a reliable guide for what comes next, given how much the actual track has diverged.

---

## Prerequisites

- Basic Python programming knowledge
- Familiarity with NumPy and basic linear algebra (helpful)
- GPU recommended from the NLP/GenAI phases onward (Phases 5–10) — Google Colab Pro is sufficient

---

## Repository Structure

```
.
├── Session 1/      # Course Intro + Python & NumPy Refresh
├── Session 2/      # Image Fundamentals
├── Session 3/      # OpenCV Basics I
├── Session 4/      # OpenCV Basics II
├── Session 5/      # Filtering & Edge Detection
├── Session 6/      # Geometric Transforms
├── Session 7/      # PyTorch Fundamentals — MLP on MNIST
├── Session 8/      # CNNs — CIFAR-10
├── Session 9/      # Transfer Learning — Cats vs Dogs
├── Session 10/      # Object Detection — YOLOv8
├── Session 11/      # Segmentation
├── Session 12/      # Pose Estimation
├── Session 13/      # OCR & Face Recognition
├── Session 14/      # Transformers & Attention (Vision)
├── Session 15/      # Video Analytics I
├── Session 16/      # Vision Transformers Lab
├── Session 17/      # Depth Estimation & 3D
├── Session 18/      # Multi-Object Tracking
├── Session 19/      # NLP & Text Processing Fundamentals
├── Session 20/      # Text Representation Fundamentals
├── Session 21/      # Word Embeddings
├── Session 22/      # Neural Networks for NLP
├── Session 23/      # Attention and Transformers (NLP)
├── Session 24/      # Language Models
├── Session 25/      # Diffusion Models & Stable Diffusion
├── Session 26/      # CLIP
├── Session 27/      # SAM — Segment Anything
├── Session 28/      # Pretraining, Fine-tuning & QLoRA
├── Session 29/      # LangChain Fundamentals
├── Session 30/      # Prompt Engineering & Chatbot Architecture
├── Session 31/      # Chatbot Labs — LangChain/Streamlit + Tools & Guardrails
├── Session 32/      # RAG Fundamentals & Vector Databases
├── Session 33/      # RAG Pipeline Lab
├── Session 34/      # Advanced & Multimodal RAG
├── Session 35/      # AI Agents & LangGraph
├── Session 36/      # Multi-Agent Systems
├── Session 37/      # LangGraph Lab II & Agentic RAG
├── Session 38/      # Model Optimisation
├── Session 39/      # FastAPI Serving & Cloud Deployment
├── Session 40/      # Vision-Language-Action Models
```

Each session folder contains whatever combination of slides (`.pptx`), a Jupyter notebook (`.ipynb`), and supporting scripts/data was actually used for that session — not every session has both slides and a notebook (see the Type / Materials columns above).

