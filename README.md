---
title: PharmaGuard Wildcard
emoji: "🧬"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🚀 PharmaGuard: Pharmacogenomic Risk Prediction System

### 🧠 Meta x Scaler OpenEnv Hackathon Submission (RIFT 2026)

PharmaGuard is an AI-powered system that analyzes patient genomic data (VCF files) to predict drug safety risks and provide personalized clinical recommendations using pharmacogenomics.

---

## 🌐 Live Demo (Hugging Face Space)

🔗 https://huggingface.co/spaces/harteij15/pharmaguard-wildcard

---

## 📓 Colab Notebook (Pipeline Demonstration)

🔗 https://colab.research.google.com/drive/1UmqC_x2E0Q7GmfU6apAv0C3ZJq2To5Pw

---

## 🎥 Demo Video

🔗 https://youtu.be/9-2n7ZQkVm0

---

## 🧬 Problem Statement

Adverse drug reactions cause over 100,000 deaths annually. Many are preventable through pharmacogenomics — understanding how genetic variants affect drug metabolism.

PharmaGuard addresses this by providing:

* Personalized drug safety analysis
* Gene-drug interaction insights
* Clinically actionable recommendations

---

## ⚙️ Features

* 📂 VCF File Parsing (Genomic Data)
* 🧬 Variant Detection using rsIDs
* 🧠 Gene → Drug Mapping (CPIC-inspired rules)
* ⚠️ Risk Classification:

  * Safe
  * Adjust Dosage
  * Toxic
  * Ineffective
* 📊 Structured Clinical JSON Output
* 📥 Downloadable Report
* 🎨 Clean Medical UI (Gradio on Hugging Face)

---

## 🏗️ How It Works

1. Upload patient VCF file
2. Extract genetic variants (rsIDs)
3. Map variants to pharmacogenomic genes
4. Infer metabolic phenotype
5. Predict drug-specific risk
6. Generate clinical recommendations + JSON report

---

## 💊 Example Use Case

**Input:**

* Drug: Morphine
* Gene: CYP2D6 (Ultrarapid Metabolizer)

**Output:**

* Risk: 🔴 Toxic
* Reason: Increased conversion leads to overdose risk

---

## 🧪 Tech Stack

* Python
* Gradio (UI)
* Hugging Face Spaces (Deployment)
* Google Colab (Pipeline Demo)

---

## 📦 Output Format

The system generates structured JSON including:

* Risk assessment
* Pharmacogenomic profile
* Clinical recommendations
* Explanation

---

## 🏆 Hackathon Alignment

* ✔ VCF Parsing & Genomic Analysis
* ✔ Drug Risk Prediction
* ✔ Explainable AI (LLM-style reasoning)
* ✔ Deployment Ready
* ✔ End-to-End Pipeline Demonstration

---

## 🚀 Future Improvements

* Integration with real CPIC databases
* Multi-drug interaction analysis
* Clinical API integration
* Enhanced LLM explanations

---

## 📌 Submission Checklist

* ✅ Live Hugging Face App
* ✅ GitHub Repository
* ✅ Colab Notebook
* ⬜ Demo Video (Pending)

---

## 🙌 Built for Impact

PharmaGuard demonstrates how AI + genomics can enable safer, personalized medicine and reduce adverse drug reactions.

---
-specific LLM systems.
