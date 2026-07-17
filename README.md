# EduRisk: Explainable AI for Transparency in Student Performance Prediction

## Overview
Student dropouts continue to be a major issue in higher education, particularly in online environments where early signs of struggling students are difficult to detect. While modern Machine Learning (ML) models offer high predictive power to identify at-risk students, they often act as “black boxes,” limiting their trust and application in real-world educational decision-making.

## The Research
To address these challenges, this study conducted a comprehensive evaluation of machine learning algorithms for binary risk classification (Pass vs. At-Risk).

The models evaluated included:

- Decision Tree (Interpretable baseline)
- XGBoost (High-performance ensemble model)

These models were first tested on a diverse benchmark from the Open University Learning Analytics Dataset (OULAD) before being adapted to a localized, institutional Moodle-based dataset. We generated Local Interpretable Model-Agnostic Explanations (LIME) to rigorously assess the explainability layer, quantitatively evaluating its fidelity and stability against the underlying black-box models.

For the related OULAD experiment repository, see: [https://github.com/noorshaumyaqilla/oulad_experiment](https://github.com/noorshaumyaqilla/oulad_experiment)

## EduRisk Checkpoint Solution & Dashboard
Building upon the research findings, this project introduces a novel approach: a three-stage Checkpoint (CP) design to account for the incremental nature of engagement and coursework data over an academic semester.

This repository contains the Proof-of-Concept Dashboard designed for real-world academic advising. The deployed dashboard utilizes:

- Algorithm: XGBoost (identified as the top-performing predictive engine)
- Data Architecture: Three adaptive checkpoints (CP1: Engagement Only, CP2: Engagement + Coursework 1, CP3: Engagement + Coursework 1 & 2)
- XAI Engine: Dual-layer explainability combining local LIME visualizations with LLM-synthesized intervention reports

This checkpoint approach ensures educators can balance high sensitivity for early warnings with high explanation fidelity for targeted, late-semester interventions. It provides stable, highly discriminative, and actionable explanations for academic advisors.

## Deployment & Setup
This application is designed to be deployed on Streamlit Cloud. It utilizes a decoupled offline-online architecture, relying on pre-processed binary files in the model_backend folder that contain pre-trained model weights, SMOTE-balanced explainer reference data, and mathematically scaled kernel widths. This overcomes memory constraints and ensures lightning-fast real-time inference without the need to retrain complex models during runtime.

### Live Demo
- Streamlit deployment: https://edurisk-dashboard.streamlit.app/

### Local Setup
1. Open the deploy_streamlit folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app locally:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL shown in the terminal.

### Deployment Notes
- Use the deploy_streamlit folder as the app root.
- Ensure requirements.txt and Procfile are present.
- The startup command should be:
  ```bash
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
  ```
