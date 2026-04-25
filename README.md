---
title: PharmaGuard Wildcard
emoji: "🧬"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# 🚀 PharmaGuard Wildcard

## 🧠 Overview
PharmaGuard Wildcard is an AI-assisted pharmacogenomic intelligence system that converts raw VCF data into medication-specific risk decisions and explainable clinical guidance. It bridges deterministic variant logic with LLM-ready outputs to create high-signal training examples for medically grounded reasoning.

By combining structured genomics parsing, genotype-to-phenotype mapping, and transparent recommendations, the project demonstrates how reliable biomedical pipelines can feed safer, more useful healthcare AI.

## 🎯 Hackathon Context
- Built for: Meta x Scaler OpenEnv Hackathon
- Theme: Wildcard (Impress Us)

This project aligns with Theme #5 by merging precision-medicine decision logic, explainability, and deployable AI interfaces into a single end-to-end system. It is both a practical clinical tooling concept and a strong OpenEnv-style data-to-intelligence pipeline.

## ⚙️ Features
- VCF ingestion and parsing for pharmacogenomically relevant variants.
- Drug-gene intelligence across high-impact medications and core PGx genes.
- Rule-based risk classification: Safe, Adjust Dosage, Toxic, Ineffective, Unknown.
- Clinically interpretable outputs: diplotype, phenotype, confidence, and recommendation blocks.
- LLM-friendly structured JSON responses for downstream model training and evaluation.
- Hugging Face Spaces-ready Gradio app entrypoint (`app.py`) for fast demos.

## 🏗️ Architecture / How It Works
1. Input layer: user uploads a VCF file or pastes VCF text.
2. Parsing layer: the engine extracts rsIDs, genes, star alleles, and genotypes.
3. Inference layer: variants are mapped to diplotype and phenotype for the selected drug's primary gene.
4. Risk layer: CPIC-aligned rule logic generates risk label, severity, and dosing direction.
5. Explanation layer: concise AI-ready narrative and structured JSON are produced as outputs.

Agent-style flow:
- Deterministic analysis agent: handles parsing + pharmacogenomic rules.
- Explainability agent: produces compact rationale for clinicians and model supervision data.

## 🚀 Quick Start
```bash
# 1) Install Python dependencies
pip install -r requirements.txt

# 2) Run the Hugging Face-compatible app
python app.py

# 3) Open the local URL shown in terminal
```

Optional Next.js interface:
```bash
npm install
npm run dev
```

## 🤗 Hugging Face Deployment
This repository is now prepared for Hugging Face Spaces (Gradio SDK):

1. Ensure these files are present at repo root:
   - `requirements.txt`
   - `app.py` (main entry file)
   - `README.md`
2. Create a new Space on Hugging Face and choose **Gradio** SDK.
3. Push this repository to the Space.
4. Spaces will auto-install dependencies from `requirements.txt` and launch `app.py`.

Recommended Space settings:
- Python: default Spaces runtime
- Hardware: CPU Basic (sufficient for this demo)
- Secrets: add API keys only if external LLM providers are enabled later

## 📊 Future Improvements
- Add multi-drug polypharmacy risk graph reasoning for interaction-aware recommendations.
- Integrate optional retrieval over CPIC/PharmGKB snippets for citation-grounded explanations.
- Add evaluation harness for LLM outputs (clinical correctness, hallucination, completeness).
- Support batch VCF ingestion for cohort-level analytics and model training pipelines.
- Add lightweight clinician feedback loops to generate preference data for RLHF-style tuning.

## 🏆 Why This Stands Out
PharmaGuard Wildcard combines real biomedical signal processing with explainable AI output design, making it more than a dashboard demo. It demonstrates practical healthcare impact, strong engineering clarity, and direct relevance to training safer domain-specific LLM systems.
