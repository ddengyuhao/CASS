# CASS: Cost-Aware Subgraph Selection for Long-Video Query Processing

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-green.svg)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()
[![Paper](https://img.shields.io/badge/Paper-ICML%202026-lightgrey)]()

**CASS** is a **training-free** framework for long-video query processing. It reformulates video token selection as a **constrained subgraph selection problem**, achieving SOTA performance on VideoMME, VRBench, and CinePile with minimal token usage.

[<a href="#-news">News</a>] • [<a href="#-abstract">Abstract</a>] • [<a href="#-framework">Framework</a>] • [<a href="#-performance">Performance</a>] • [<a href="#-installation">Installation</a>] • [<a href="#-usage">Usage</a>] • [<a href="#-citation">Citation</a>]

</div>

## 📖 Abstract

As large multimodal models (LMMs) become the dominant approach for video analysis, managing the massive volume of visual tokens in long-form videos has become a critical data management challenge. Naively feeding full-length video streams into LMMs results in excessive token usage. 
To overcome this, we propose CASS (Cost-Aware Subgraph Selection), a training-free framework for long-video query processing. Instead of treating video as a flat sequence of tokens, CASS reformulates efficient query processing fundamentally as a budget-aware video token selection problem. By modeling video content as a structured event graph, we execute this token selection as a constrained subgraph routing task, where the selected subgraph serves as an explicit query plan under a fixed token budget.
CASS consists of three components: (1) an offline semantic indexing phase that constructs a directed event graph capturing both temporal and long-range dependencies; (2) a budget-aware query planning module that selects an informative subgraph using a monotone submodular objective with theoretical guarantees; and (3) a graph-guided execution pipeline that enforces structured reasoning over validated graph paths to improve answer correctness and reduce hallucination.
 Extensive experiments across multiple long-video benchmarks demonstrate that, under the same token budget, CASS yields an average accuracy improvement of up to 6.7\% over the SOTA baseline, while substantially reducing visual token usage by over 90\% compared to full-sequence inference.

---

## 🛠️ Framework

> **Figure 1**: The overall framework of CASS. We first decompose the video into events, construct a weighted graph capturing semantic and temporal links, and then select a budget-constrained subgraph to guide the LMM's reasoning.

<div align="center">
  <img src="assets/framework.png" width="95%" alt="CASS Framework"/>
  <br>
</div>

### Key Components

1.  **Offline Semantic Indexing (Graph Construction)**: 
    * Decomposes video into events using **TransNetV2**.
    * Extracts global ([CLS]) and local patch features using **CLIP**.
    * Builds a graph $G=(V, E)$ with **temporal edges** (sequential flow) and **semantic edges** (long-range similarity).
2.  **Budget-Aware Query Planning (Subgraph Selection)**: 
    * Solves the submodular optimization problem.
    * The objective function balances **Query Relevance** and **Reachable Information Gain**.
    * Uses the **CELF algorithm** for fast, near-optimal selection.
3.  **Graph-Guided Execution Pipeline (Answer Synthesis):**:
    * Guides the LMM to verify evidence and propagate logic strictly along the selected graph paths.

---

## 📊 Performance

We evaluate CASS on **VideoMME**, **VRBench**, and **CinePile**. Under a strict budget of **8,192 visual tokens**, our method significantly outperforms existing efficient inference baselines.

| Method | Type | LLM Backbone | VideoMME | VRBench | CinePile | Avg. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| *Proprietary LMMs* | | | | | | |
| Gemini 1.5 Pro | API | - | 75.0 | 70.7 | 60.1 | 68.6 |
| GPT-4o | API | - | 71.9 | 68.7 | 56.1 | 65.6 |
| *Open-Source LMMs (Full-Sequence)* | | | | | | |
| InternVL2.5-78B | Dense | - | 72.1 | 53.5 | 54.6 | 60.1 |
| Qwen2-VL-72B | Dense | - | 71.2 | 59.1 | 54.2 | 61.5 |
| Qwen2.5-VL-7B | Dense | - | 65.1 | 56.5 | 52.6 | 58.1 |
| LLaVA-NeXT-34B | Dense | - | 70.6 | 48.5 | 41.5 | 53.5 |
| *Efficient Strategies (Qwen2.5-VL-7B)* | | | | | | |
| LLaVA-Phi | Architecture | Phi-2 | 34.5 | 30.0 | 27.5 | 30.7 |
| FastV | Token Reduction | Qwen2.5-VL-7B | 52.3 | 43.1 | 38.4 | 44.6 |
| DyCoke | Token Reduction | Qwen2.5-VL-7B | 51.8 | 47.2 | 37.1 | 45.4 |
| VTR-VLM | Token Reduction | Qwen2.5-VL-7B | 54.1 | 53.3 | 44.5 | 50.6 |
| AdaReTaKe | Token Reduction | Qwen2.5-VL-7B | 50.8 | 47.1 | 46.4 | 48.1 |
| Q-Frame | Keyframe Sampling | Qwen2.5-VL-7B | 53.5 | 48.5 | 40.7 | 47.6 |
| MovieChat | Memory-based | Qwen2.5-VL-7B | 48.5 | 35.0 | 28.0 | 37.2 |
| SGVC | Caption-based | Qwen2.5-VL-7B | 45.0 | 32.0 | 30.2 | 35.7 |
| **Ours** | **Graph-based** | **Qwen2.5-VL-7B** | **61.5** | **54.8** | **48.1** | **54.8** |
| *Efficient Strategies (Qwen2-VL-72B)* | | | | | | |
| FastV | Token Reduction | Qwen2-VL-72B | 56.5 | 45.5 | 41.6 | 47.9 |
| DyCoke | Token Reduction | Qwen2-VL-72B | 57.9 | 46.8 | 39.8 | 48.2 |
| VTR-VLM | Token Reduction | Qwen2-VL-72B | 58.2 | 55.4 | 45.5 | 53.0 |
| AdaReTaKe | Token Reduction | Qwen2-VL-72B | 55.5 | 48.2 | 48.0 | 50.6 |
| Q-Frame | Keyframe Sampling | Qwen2-VL-72B | 62.0 | 52.1 | 41.9 | 52.0 |
| MovieChat | Memory-based | Qwen2.5-VL-7B | 52.1 | 37.5 | 28.5 | 39.4 |
| SGVC | Caption-based | Qwen2.5-VL-7B | 49.5 | 34.8 | 31.1 | 38.5 |
| **Ours** | **Graph** | **Qwen2-VL-72B** | **69.2** | **58.5** | **51.3** | **59.7** |


> *Results sourced from Table 1 in our paper.*

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/ddengyuhao/CASS.git](https://github.com/ddengyuhao/CASS.git)
cd CASS

```

### 2. Environment Setup

We recommend using Conda to manage dependencies.

```bash
conda create -n cass python=3.10 -y
conda activate cass

# Install core dependencies
pip install -r requirements.txt

# Install TransNetV2 (for shot detection) and Decord (for video loading)
pip install transnetv2-pytorch decord

```

---

## 📂 Data Preparation

Please organize your datasets as follows. You can configure the `DATA_ROOT` in `scripts/run.sh`.

```text
dataset/
├── VideoMME/
│   ├── videos/       # Contains .mp4 files
│   └── test.json     # Annotation file (Hardcoded in code)
├── CinePile/
│   ├── yt_videos/    # Downloaded video files
│   └── cookies.txt   # (Optional) YouTube auth for downloading restricted videos
└── VRBench/
    ├── videos/       # Contains video files
    └── VRBench.json  # Annotation file
```

---

## 🏃 Usage

You can run inference using the provided shell script, which supports multi-GPU chunking.

### Quick Start

To evaluate on **VideoMME** using **Qwen2.5-VL-7B**:

```bash
bash scripts/run.sh

```

### Advanced Configuration

You can also run the Python script directly for specific configurations:

```bash
python scripts/run_inference.py \
    --dataset VideoMME \
    --data_root ./dataset \
    --backbone Qwen2.5-VL-7B \
    --method CASS \
    --token_budget 8192 \
    --tau 30.0 \
    --delta 0.65 \
    --output_dir ./results/debug

```

#### Key Arguments:

* `--method`: Selection strategy (Default: `CASS`).
* `--backbone`: Model backbone (e.g., `Qwen2.5-VL-7B`, `Qwen2-VL-72B`).
* `--token_budget`: Maximum number of visual tokens allowed (default: 8192).
* `--tau`: Temporal distance threshold for graph construction (default: 30.0).
* `--delta`: Semantic similarity threshold (default: 0.65).



---

## 📝 Citation

If you find this project useful for your research, please cite our paper:

```bibtex
@article{cass2026,
  title={CASS: Cost-Aware Subgraph Selection for Long-Video Query Processing},
  author={Yuhao Deng, Hanqing Hu, Chengliang Chai, Qiyan Deng, Feicheng Li, Ying Nie, Zhiming Peng, Daiting Shi, Ye Yuan, Guoren Wang},
  journal={Under Review at VLDB},
  year={2027}
}

```
