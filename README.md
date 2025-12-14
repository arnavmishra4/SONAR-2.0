# 🏛️ SONAR 2.0: Multi-Model Ensemble for Archaeological Site Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace]([https://img.shields.io/badge/🤗-Demo-yellow.svg](https://huggingface.co/spaces/arnavmishra4/SONAR?logs=container))](YOUR_HUGGINGFACE_SPACE_LINK)

> **Automated detection of archaeological sites from multi-source geospatial data using deep learning ensemble with GATE meta-learner**

![Banner](https://drive.google.com/uc?export=view&id=1wBa_k9YPdpxKBSp85iqtVq07bW1yXDhW)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 Overview

**SONAR 2.0** is an advanced machine learning pipeline designed to detect potential archaeological sites from remotely sensed geospatial data. By combining **LiDAR terrain data**, **Sentinel-2 satellite imagery**, and **hydrological flow data**, the system identifies anomalous terrain patterns that may indicate buried or surface archaeological features.

### Why SONAR 2.0?

Traditional archaeological surveys are:
- ⏱️ **Time-consuming**: Manual field surveys can take months
- 💰 **Expensive**: Requires specialized equipment and personnel
- 🌍 **Limited in scope**: Can only cover small areas

SONAR 2.0 enables:
- ⚡ **Rapid screening** of large geographic areas
- 🎯 **Focused fieldwork** by identifying high-probability sites
- 🔍 **Discovery of hidden sites** obscured by vegetation or terrain

### Real-World Applications

- 🏺 **Preventive archaeology**: Identify sites before construction projects
- 🗺️ **Heritage management**: Map cultural heritage at landscape scale
- 🌳 **Remote area surveys**: Detect sites in inaccessible rainforest/desert regions
- 📊 **Research**: Understand settlement patterns and landscape archaeology

---

## ✨ Key Features

### Multi-Source Data Integration
- **LiDAR DTM** (Digital Terrain Model): High-resolution elevation data revealing subtle topographic features
- **Sentinel-2 Imagery**: NDVI (vegetation) and NDWI (water) indices for environmental context
- **HydroSHEDS**: Flow accumulation and direction for understanding ancient water management

### Advanced AI Architecture
- **4-Model Ensemble**: Combines complementary anomaly detection approaches
- **GATE Meta-Learner**: Intelligent fusion of model predictions
- **Archaeological Similarity**: Compares patches against 18 verified archaeological reference sites
- **5-Channel Output**: Comprehensive probability maps for each model + final prediction

### Production-Ready Pipeline
- ⚙️ **Fully automated**: From raw data to predictions
- 📦 **Batch processing**: Handle hundreds of AOIs efficiently
- 🔧 **Configurable**: Easy customization via config files
- 📊 **Scalable**: Processes 50+ patches/second on GPU

---

## 🏗️ System Architecture

### Complete Pipeline Flow

![System Architecture](https://drive.google.com/file/d/1wBa_k9YPdpxKBSp85iqtVq07bW1yXDhW/view?usp=sharing)

---

## 🔬 How It Works

### Stage 1: Data Preparation

The system ingests three types of geospatial data:

1. **LiDAR DTM** (Primary source)
   - High-resolution elevation data (typically 1-5m resolution)
   - Reveals subtle topographic features like mounds, ditches, platforms
   - Preprocessed to extract:
     - **Slope**: Rate of elevation change
     - **Roughness**: Surface texture variability

2. **Sentinel-2 Multispectral Imagery**
   - **NDVI** (Normalized Difference Vegetation Index): Identifies vegetation patterns that may indicate buried structures
   - **NDWI** (Normalized Difference Water Index): Detects ancient water management features

3. **HydroSHEDS Hydrological Data**
   - **Flow Accumulation**: Historical water flow patterns (log-transformed)
   - **Flow Direction**: Encoded as sine for circular continuity

**Result**: 7-channel patches (64×64 pixels) with comprehensive terrain + environmental context

---

### Stage 2: Multi-Model Ensemble

#### **Model 1: ResUNet Autoencoder**
```
Purpose: Detect anomalies through reconstruction error
Architecture: U-Net with residual blocks
Input: 7-channel patch → 256-dim latent space → 7-channel reconstruction
Anomaly Signal: Patches that are difficult to reconstruct (high error) are anomalous
```

**Why it works**: Archaeological features create unusual patterns that differ from natural terrain, resulting in higher reconstruction errors.

---

#### **Model 2: Isolation Forest**
```
Purpose: Identify outliers in latent embedding space
Input: 128-dim patch embeddings (from trained encoder)
Method: Ensemble of 100 decision trees
Contamination: 5% (expected proportion of anomalies)
```

**Why it works**: Archaeological sites form distinct clusters in feature space, separate from natural landscape patterns.

---

#### **Model 3: K-Means Clustering**
```
Purpose: Match patches against known archaeological signatures
Input: 128-dim patch embeddings
Clusters: 10 representative terrain patterns
Scoring: Cosine similarity to cluster centroids
```

**Why it works**: Archaeological sites share common morphological signatures (e.g., rectangular enclosures, circular mounds) that can be learned from examples.

---

### Stage 3: PRE-GATE Layer

Before final prediction, the system computes an additional **Archaeological Similarity Score**:

```python
# Pseudocode
for each patch:
    embedding = encoder.extract(patch)  # 128-dim vector
    
    similarities = []
    for reference_site in 18_verified_archaeological_sites:
        sim = cosine_similarity(embedding, reference_site.embedding)
        similarities.append(sim)
    
    arch_similarity_score = max(similarities)
```

This creates a **4-feature vector** for GATE:
- `prob1`: Autoencoder anomaly score
- `prob2`: Isolation Forest outlier score  
- `prob3`: K-Means cluster similarity
- `sim_score`: Archaeological reference similarity

---

### Stage 4: GATE Meta-Learner

**GATE** (Gated Aggregate Transformer Ensemble) is a lightweight MLP that learns optimal weights for combining model predictions:

```
Architecture:
Input Layer:    4 features (prob1, prob2, prob3, sim_score)
Hidden Layer 1: 16 neurons (ReLU)
Hidden Layer 2: 8 neurons (ReLU)
Output Layer:   1 neuron (Sigmoid) → Binary classification
```

**Training**: Supervised learning on labeled examples (archaeological vs. non-archaeological patches)

**Inference**: 
- Output > 0.5 → Archaeological site candidate (label: 1)
- Output ≤ 0.5 → Normal terrain (label: 0)

---

### Stage 5: Output Generation

For each AOI, the system generates:

1. **Unified Probability Matrix** (5 channels per patch)
   - Shape: `(num_patches, 64, 64, 5)`
   - Each channel represents one model's confidence

2. **Binary Prediction Map**
   - Full AOI heatmap showing GATE decisions

3. **Top Candidates List**
   - Ranked patches by GATE confidence score

4. **Statistics Report**
   - Percentage of positive predictions
   - Mean confidence scores
   - Distribution analysis

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- 16GB RAM (minimum)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/SONAR-2.0-Archaeological-Detection.git
cd SONAR-2.0-Archaeological-Detection
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n sonar python=3.8
conda activate sonar
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

```bash
# Option 1: Download from GitHub Releases
# Visit: https://github.com/yourusername/SONAR-2.0/releases
# Download models.zip and extract to models/

# Option 2: Download from HuggingFace
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='yourusername/sonar-2.0-models', local_dir='models/')"
```

### Step 5: Configure Paths

```bash
cp .env.example .env
# Edit .env with your data paths
```

Example `.env`:
```bash
HYDRO_DATA_PATH=/data/hydroshed/
SENTINEL_DATA_PATH=/data/sentinel2/
LIDAR_DATA_PATH=/data/lidar_dtm/
```

---

## ⚡ Quick Start

### Option 1: Run Inference with Pre-trained Models

If you already have prepared patch data:

```bash
# Generate predictions for AOIs 0-10
python src/main.py --mode predict --aoi-start 0 --aoi-end 10

# Output will be saved to:
# data/unified_probability_matrices_with_gate/
```

### Option 2: Complete Pipeline from Raw Data

```bash
# Step 1: Prepare dataset (extracts 7-channel patches)
python src/main.py --mode prepare

# Step 2: Generate predictions
python src/main.py --mode predict

# Or run both in one command:
python src/main.py --mode full
```

### Option 3: Train Your Own Models

```bash
# Train all models from scratch
python src/train.py --all

# Or train individual models:
python src/train.py --model autoencoder --epochs 50
python src/train.py --model iforest
python src/train.py --model kmeans
```

---


## 🤖 Model Details

### Model 1: ResUNet Autoencoder

| Property | Value |
|----------|-------|
| **Architecture** | Residual U-Net with skip connections |
| **Input** | 7 channels × 64×64 pixels |
| **Latent Space** | 256 dimensions (bottleneck) |
| **Parameters** | ~5.2M trainable |
| **Loss Function** | MSE (Mean Squared Error) |
| **Training Time** | ~8 hours (RTX 3090) |

**Key Innovation**: Uses sigmoid uncertainty scoring instead of raw MSE for better anomaly calibration.

---

### Model 2: Isolation Forest

| Property | Value |
|----------|-------|
| **Algorithm** | Ensemble of isolation trees |
| **Input** | 128-dim patch embeddings |
| **Trees** | 100 |
| **Contamination** | 5% (configurable) |
| **Training Time** | ~30 minutes (CPU) |

**How it works**: Anomalies are easier to isolate (require fewer splits in decision trees).

---

### Model 3: K-Means Clustering

| Property | Value |
|----------|-------|
| **Algorithm** | K-Means++ initialization |
| **Input** | 128-dim patch embeddings |
| **Clusters** | 10 (representing terrain archetypes) |
| **Distance Metric** | Euclidean → Cosine similarity |
| **Training Time** | ~20 minutes (CPU) |

**Purpose**: Learn typical archaeological terrain patterns from training data.

---

### Model 4: GATE Meta-Learner

| Property | Value |
|----------|-------|
| **Architecture** | MLP (4 → 16 → 8 → 1) |
| **Input** | 4 aggregated features |
| **Activation** | ReLU (hidden), Sigmoid (output) |
| **Parameters** | ~500 |
| **Training Time** | ~5 minutes (CPU) |
| **Loss Function** | Binary Cross-Entropy |

**Training Data Required**: Labeled examples of archaeological vs. non-archaeological patches.

---
---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: Share your ideas for improvements
3. **Submit Pull Requests**: 
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/AmazingFeature`)
   - Commit changes (`git commit -m 'Add AmazingFeature'`)
   - Push to branch (`git push origin feature/AmazingFeature`)
   - Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📚 Citation

If you use SONAR 2.0 in your research, please cite:

```bibtex
@software{sonar2_2024,
  author = {Your Name},
  title = {SONAR 2.0: Multi-Model Ensemble for Archaeological Site Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/SONAR-2.0-Archaeological-Detection},
  note = {Deep learning pipeline for detecting archaeological sites from geospatial data}
}
```

### Related Publications

*Add any papers, theses, or conference presentations here*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Data Sources

- **LiDAR Data**: [NASA GEDI](https://gedi.umd.edu/) / [OpenTopography](https://opentopography.org/)
- **Sentinel-2**: [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- **HydroSHEDS**: [WWF / USGS](https://www.hydrosheds.org/)

Please respect the licenses of these data sources.

---

## 🙏 Acknowledgments

- **Archaeological Domain Experts**: For validating reference sites and providing ground truth
- **Open Source Community**: PyTorch, Scikit-learn, Rasterio contributors
- **Compute Resources**: [Institution/Cloud provider if applicable]

---

## 📧 Contact

**Your Name**  
🎓 [Your University/Organization]  
📧 Email: your.email@example.com  
🔗 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
🐦 Twitter: [@yourhandle](https://twitter.com/yourhandle)  
💼 Portfolio: [yourwebsite.com](https://yourwebsite.com)

---

## 🔗 Links

- 🤗 **[Live Demo on HuggingFace](YOUR_HUGGINGFACE_SPACE_LINK)** - Try it yourself!
- 📦 **[Model Weights on HuggingFace Hub](YOUR_MODEL_HUB_LINK)**
- 📊 **[Project Paper/Thesis](YOUR_PAPER_LINK)** *(if available)*
- 🎥 **[Demo Video](YOUR_VIDEO_LINK)** *(if available)*

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ for archaeological heritage preservation

</div>
