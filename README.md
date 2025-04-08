<div id="top" align="center">

# Neural Modeling Methods & Tools

**Advanced Neural Network Implementations and Research Projects**

[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Projects](#-projects)
  - [Centerline Detection](#centerline-detection)
  - [Deep Neural Networks](#deep-neural-networks)
  - [Recurrent Neural Networks](#recurrent-neural-networks)
  - [PyTorch Models](#pytorch-models)
- [🚗 OpenLane-V2 Integration](#-openlane-v2-integration)
- [Implementations](#-implementations)
- [Model Performance](#-model-performance)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Citation](#-citation)
- [Related Resources](#-related-resources)

## 🔭 Overview

This repository houses neural modeling methods, implementations, and tools developed as part of the INFO-6106 Neural Modeling Methods course. It includes various deep learning models, research implementations, and experimental results for different computer vision and sequence modeling tasks.

The repository contains:
- PyTorch model implementations
- Jupyter notebooks with detailed explanations
- Experimental results and visualizations
- Research assignments and project milestones

## 📊 Projects

<details>
<summary><b>Centerline Detection</b></summary>
<br>

### Centerline Detection

Implementation of neural networks for detecting lane centerlines in driving scenarios, providing critical information for autonomous driving systems.

**Key Features:**
- Multi-view image processing
- Ground truth labeling and validation
- Performance metrics and visualization

</details>

<details>
<summary><b>Deep Neural Networks</b></summary>
<br>

### Deep Neural Networks

Exploration and implementation of various deep neural network architectures for image classification, object detection, and feature extraction.

**Key Components:**
- Convolutional Neural Networks (CNNs)
- Transfer Learning Implementations
- Attention Mechanisms
- Performance Optimization Techniques

</details>

<details>
<summary><b>Recurrent Neural Networks</b></summary>
<br>

### Recurrent Neural Networks

Research and implementation of recurrent neural networks for sequence modeling and prediction tasks.

**Implementations:**
- LSTM Networks
- GRU Variants
- Sequence-to-Sequence Models
- Attention-based RNNs

</details>

<details>
<summary><b>PyTorch Models</b></summary>
<br>

### PyTorch Models

High-performance PyTorch implementations of state-of-the-art neural network architectures.

**Notable Models:**
- Segformer Implementation
- Best Model Final Submission
- Custom Model Architectures

</details>

<details>
<summary><b>OpenLane-V2 Integration</b></summary>
<br>

### OpenLane-V2 Integration

Integration with the OpenLane-V2 project, a comprehensive perception and reasoning benchmark for scene structure in autonomous driving.

<div align="center">
  <a href="https://github.com/OpenDriveLab/OpenLane-V2/blob/master/README.md#introducing-openlane-v2-update">
    <img src="https://img.shields.io/badge/OpenLane--V2-Integration-orange?style=for-the-badge" alt="OpenLane-V2">
  </a>
</div>

**Key Features:**
- **Lane Segment Representation**: A unifying approach for comprehensive scene understanding
- **SD Map Integration**: Standard-definition maps providing topological and positional priors
- **3D Lane Detection**: Advanced spatial representation of lane structures
- **Topology Reasoning**: Understanding relationships between lanes and traffic elements

<p align="center">
  <img src="https://github.com/OpenDriveLab/OpenLane-V2/assets/29263416/77846f69-fe77-45aa-b769-e85fd98a0596" width="500px" alt="OpenLane-V2 Lane Segment Visualization">
</p>

#### Lane Segment Functionality

Lane segment representation offers comprehensive functionality:

<table align="center">
  <tr align="center">
    <td><b>Feature</b></td>
    <td><b>Capability</b></td>
  </tr>
  <tr align="center">
    <td>3D Space Representation</td>
    <td>✅</td>
  </tr>
  <tr align="center">
    <td>Lane Direction</td>
    <td>✅</td>
  </tr>
  <tr align="center">
    <td>Lane-level Drivable Area</td>
    <td>✅</td>
  </tr>
  <tr align="center">
    <td>Lane-lane Topology</td>
    <td>✅</td>
  </tr>
  <tr align="center">
    <td>Traffic Element Integration</td>
    <td>✅</td>
  </tr>
</table>

<div align="center">
  <a href="https://github.com/OpenDriveLab/OpenLane-V2/blob/master/README.md#introducing-openlane-v2-update" target="_blank">
    <button style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px; border: none;">
      Learn More About OpenLane-V2
    </button>
  </a>
</div>

</details>

## 💻 Implementations

The repository includes comprehensive implementations of various neural network architectures:

| Project | Type | Description |
|---------|------|-------------|
| Centerline Detection | Computer Vision | Neural networks for detecting road centerlines |
| DNN | Deep Learning | Various deep neural network architectures |
| RNN | Sequence Modeling | Recurrent neural networks for sequential data |
| YOLO | Object Detection | Implementation with smaller datasets |
| Segformer | Semantic Segmentation | PyTorch implementation of Segformer |
| OpenLane-V2 | Autonomous Driving | Integration with lane segment representation |
| Research Assignment 3 | Research | Neural research implementations |

## 📈 Model Performance

Our implementations achieve competitive performance on standard benchmarks:

- **YOLO with Smaller Dataset**: Demonstrated effective object detection with limited training data
- **Segformer PyTorch**: Achieved high accuracy in semantic segmentation tasks
- **RNN Implementations**: Effective sequence modeling and prediction capabilities
- **OpenLane-V2 Integration**: Enhanced lane detection with topology reasoning

## 🛠️ Technologies Used

This repository leverages several key technologies:

- **PyTorch**: Deep learning framework
- **Jupyter Notebooks**: Interactive development and visualization
- **NumPy/Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **CUDA**: GPU acceleration for model training
- **Git LFS**: Version control for large model files
- **OpenLane-V2**: Autonomous driving scene topology benchmark

## 🚀 Getting Started

To get started with this repository:

```bash
# Clone the repository
git clone https://github.com/Omsatyaswaroop29/INFO-6106--Neural-Modeling-Methods-Tool-.git

# Navigate to the repository
cd INFO-6106--Neural-Modeling-Methods-Tool-

# Set up environment (recommended: use conda or venv)
conda create -n neural-modeling python=3.8
conda activate neural-modeling

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook
```

## 📝 Citation

If you find this repository useful for your research, please use the following citation:

```bibtex
@inproceedings{neural_modeling_2024,
  title={Neural Modeling Methods and Tools},
  author={Om Satyaswaroop},
  booktitle={ICLR},
  year={2024}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## 🔗 Related Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2/blob/master/README.md#introducing-openlane-v2-update) | [DriveAGI](https://github.com/OpenDriveLab/DriveAGI) | [DriveLM](https://github.com/OpenDriveLab/DriveLM) 
- [OpenScene](https://github.com/OpenDriveLab/OpenScene) | [TopoNet](https://github.com/OpenDriveLab/TopoNet) | [LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet)
- [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane) | [OpenLane](https://github.com/OpenDriveLab/OpenLane)
- [BEV Perception Survey & Recipe](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [BEVFormer](https://github.com/fundamentalvision/BEVFormer)

<p align="right">(<a href="#top">back to top</a>)</p>
