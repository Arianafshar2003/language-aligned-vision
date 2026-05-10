# Scene Relational Representations in High-Level Visual Cortex

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for: **"Language-aligned models and structured scene descriptions reveal relational representations in high-level visual cortex"**

Rajaei, K., Afshar, A., & Soltanian-Zadeh, H. (2025)

## Overview

This repository implements voxelwise encoding models and variance partitioning analyses to test whether high-level visual cortex represents relational scene structure. Using 7T fMRI data from the Natural Scenes Dataset (NSD), we compare structured narrative descriptions with reduced lexical controls and language-aligned vision models with vision-only models.

## Pipeline Overview

1. **Beta Preparation** - Extract and average single-trial fMRI responses
2. **Scene Annotation** - Generate structured descriptions using Qwen2.5-VL
3. **Text Feature Extraction** - CLIP ViT-L/14 and RoBERTa-base embeddings
4. **Visual Feature Extraction** - DINOv3 and SigLIP2 embeddings
5. **Voxelwise Encoding** - Ridge regression with cross-validated alpha selection
6. **Variance Partitioning** - Unique variance analysis with bootstrap significance
7. **Image-wise Analysis** - Stimulus-level unique contribution scores

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- NSD dataset access (https://naturalscenesdataset.org/)

## Installation

git clone https://github.com/arianafshar2003/language-aligned-vision.git
cd ipm_final
pip install -r requirements.txt

## Contact
- **Arian Afshar** — [aryanafshar1382@gmail.com](mailto:aryanafshar1382@gmail.com) / [arian.afshar@ipm.ir](mailto:arian.afshar@ipm.ir)
