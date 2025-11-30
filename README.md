# Soft-Attention Skin Cancer Classification (HAM10000) 🩺🧪

This repository contains my experiments on **skin lesion classification** using the  
**HAM10000 dataset** and **soft-attention–based CNNs**, inspired by the paper:

> **Soft-Attention Improves Skin Cancer Classification Performance**  
> Soumyya Kanti Datta et al. (2021)  
> arXiv: [2105.03358](https://arxiv.org/abs/2105.03358)

The goal is to explore whether adding a **soft-attention module** on top of standard
CNN backbones (e.g., ResNet, VGG, Inception-like models) improves classification
performance on HAM10000 under **limited compute**.

---

## ✨ Project Highlights

- Reproduction attempt of *Soft-Attention Improves Skin Cancer Classification Performance* on the **HAM10000** dataset.
- Experiments primarily implemented in **Jupyter Notebooks** (`notebooks/`).
- Uses **DVC** for experiment/data tracking (`.dvc/`, `.dvcignore`).
- Experiment helper utilities live in **`exp_tools/`** (training, evaluation, logging, etc.).
- Supports Multi-GPU training with PyTorch’s `DistributedDataParallel` (if available).

> ⚠️ Note: Due to resource limitations, many experiments are **partial or exploratory** rather than full-scale reproductions of the paper.

---

## 📁 Repository Structure

At a high level, the repo is organized as:

```text
.
├── .dvc/               # DVC internal metadata (data/experiments not committed)
├── .dvcignore
├── .gitignore
├── exp_tools/          # Python utilities for experiments (training loops, configs, etc.)
├── notebooks/          # Main work: EDA, training, attention modules, evaluation
├── pyproject.toml      # Project metadata and dependencies
└── README.md           # (this file)
```

> The **actual logic** (models, preprocessing, evaluation) is inside the `exp_tools` package, the notebook uses it to perform experiments.  

You can open the notebooks in order (usually EDA → preprocessing → training → evaluation).

---

## 🧬 Dataset: HAM10000

The project uses the **HAM10000** ("Human Against Machine with 10,000 training images") dataset of dermatoscopic images.  
You can download it from the ISIC archive or Kaggle (for example):

- HAM10000 on Kaggle: <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>

The dataset contains 10,015 images across 7 classes, typically abbreviated as:

- `akiec` – Actinic keratoses
- `bcc`   – Basal cell carcinoma
- `bkl`   – Benign keratosis-like lesions
- `df`    – Dermatofibroma
- `nv`    – Melanocytic nevi
- `mel`   – Melanoma
- `vasc`  – Vascular lesions

In the notebooks I handle:

- Class imbalance (especially `nv` vs rare classes)
- Basic cleaning / preprocessing
- Resizing + normalization for CNN backbones

---

## ⚙️ Environment & Installation

This repository uses a `pyproject.toml` for dependency management.

### 1. Clone the repo

```bash
git clone https://github.com/obsessor-ak1/Skin_Cancer_Detection_HAM10000.git
cd Skin_Cancer_Detection_HAM10000
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# or
.venv\Scripts\Activate.ps1         # Windows (Powershell)
```

### 3. Install dependencies

You have two options depending on how you like to work:

**Option A – Install as a local project (if build system is configured):**

```bash
pip install -e .
```

**Option B – Install from `pyproject.toml` manually:**

Open `pyproject.toml` and install the listed dependencies, for example:

```bash
pip install torch torchvision
pip install jupyter
pip install numpy pandas matplotlib scikit-learn
pip install dvc
# + any other libraries listed in pyproject.toml
```

---

## 🚀 How to Run the Experiments

This project is notebook-centric.

1. Launch Jupyter `jupyter notebook`
2. Open the notebooks in `notebooks/`


## 🧠 Soft-Attention Idea (Short Summary)

The core idea from the paper (and this repo) is to add a **soft-attention block** to CNN feature maps:

- A small network produces **K attention maps** over spatial locations.
- These attention maps are normalized (softmax-like) and used to **reweight** the feature maps.
- This boosts regions important for the lesion and suppresses background artifacts (veins, hair, etc.).
- It also improves **interpretability**, since attention heatmaps can be visualized over the image.

---

## 📊 Results (Partial / Work-in-Progress)

Because this project was run on limited hardware:

- Experiments are **not full reproductions** of the paper.
- Results so far are **exploratory**.

You can:

- Check the final cells of the training/evaluation notebooks for:
  - Training curves (loss/accuracy)
  - Per-class precision/recall/F1
  - Confusion matrices
  - Example attention heatmaps

If you run the same pipeline on a stronger GPU and full dataset for enough epochs, you should be able to get closer to the performance reported in the original paper.

---

## 🛠️ Future Work / TODO

Some things I’d like to improve (or that you can contribute):

- Clean, modular **PyTorch training script(s)** outside notebooks.
- Config-driven experiments (YAML/JSON configs for model, data, training hyperparams).
- Full **attention-augmented versions** of multiple other backbones (ResNet, Inception-ResNet, DenseNet).
- Better handling of **class imbalance** (focal loss, class weights, advanced augmentations).
- Reproduce (or beat!) the metrics from the original paper on HAM10000.
- Implement other better papers or approaches of skin cancer classification on this dataset.

---

## 📚 References

- Soumyya Kanti Datta et al.,  
  **Soft-Attention Improves Skin Cancer Classification Performance** (2021), arXiv:2105.03358.  
- HAM10000 Dataset –  
  Tschandl et al., *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*.
- ISIC Archive – International Skin Imaging Collaboration: <https://www.isic-archive.com/>
- Official Keras Implementation: [Attention Based Skin Cancer Classification](https://github.com/skrantidatta/Attention-based-Skin-Cancer-Classification.git)
---

## 👤 Author

**Aayush Kumar**  
GitHub: [@obsessor-ak1](https://github.com/obsessor-ak1)

> This repo is mainly for my own learning and experimentation,  
> but feel free to open issues or suggest improvements.
