# Multi-Modal-Image-Sentiment-Analysis

## Overview
This PR introduces a new **multidimensional 3D CNN model** within the Onigiri project.  
The model leverages a large-scale dataset (~18TB) capturing regressional relationships between **mood, emotion, and facial expressions**, along with **gender attributes**.  

The goal is to extend the multimodal project by enabling **mood determination** from image and face data, integrated with contextual metadata.

---

## Key Features
- **New Data Integration**
  - Added ~18TB of mass data on mood, emotion, and facial expression alongside gender.
  - Preprocessing pipeline supports sequence-based image and embedding fusion.

- **3D CNN Model Implementation**
  - Input: `data_input` (sequence of facial image tensors).
  - Auxiliary Input: `site_id_input` for contextual weather embedding.
  - Weather embedding reshaped into a **weather map** and concatenated as an additional channel.
  - Temporal-spatial Conv3D layers with ELU activations.
  - Dense fully connected layers leading to mood prediction outputs.

- **Output**
  - Predicts **mood state** given image and contextual inputs.
  - Designed to integrate seamlessly with existing multimodal architecture.

---

## Motivation
This implementation expands Onigiri’s capability:
- Moves beyond **basic sentiment analysis** to deeper **mood-level understanding**.
- Bridges the gap between **visual emotion recognition** and **context-aware multimodal inference**.
- Scales to massive datasets, aligning with the multimodal project’s growth roadmap.

---

## Next Steps
- Train and benchmark the new model on curated dataset splits.
- Compare performance against existing CNN and multimodal baselines.
- Integrate evaluation metrics for mood detection accuracy and generalization.

---
