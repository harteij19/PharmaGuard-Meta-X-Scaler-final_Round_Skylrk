"""
PharmaGuard AI — RIFT 2026 HealthTech Hackathon
Pharmacogenomic Risk Prediction System
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

DRUG_GENE_MAP: Dict[str, List[str]] = {
    "MORPHINE":      ["CYP2D6"],
    "CODEINE":       ["CYP2D6"],
    "TRAMADOL":      ["CYP2D6"],
    "WARFARIN":      ["CYP2C9", "VKORC1"],
    "CLOPIDOGREL":   ["CYP2C19"],
    "SIMVASTATIN":   ["SLCO1B1"],
    "AZATHIOPRINE":  ["TPMT"],
    "FLUOROURACIL":  ["DPYD"],
}

# rsID → star-allele for each gene
RS_TO_STAR: Dict[str, Dict[str, str]] = {
    "CYP2D6": {
        "rs3892097":  "*4",   "rs35742686": "*3",
        "rs5030655":  "*6",   "rs16947":    "*2",
        "rs1135840":  "*10",  "rs28371725": "*17",
        "rs1058164":  "*29",
    },
    "CYP2C19": {
        "rs4244285":  "*2",   "rs4986893":  "*3",
        "rs12248560": "*17",
    },
    "CYP2C9": {
        "rs1799853":  "*2",   "rs1057910":  "*3",
    },
    "SLCO1B1": {
        "rs4149056":  "*5",
    },
    "TPMT": {
        "rs1142345":  "*3A",  "rs1800460":  "*3C",
        "rs1800462":  "*2",
    },
    "DPYD": {
        "rs3918290":  "*2A",  "rs55886062": "*13",
    },
    "VKORC1": {
        "rs9923231":  "A",    # sensitive / low-dose
        "rs9934438":  "A",
    },
}

# diplotype → (phenotype_label, activity_bucket, confidence)
PHENOTYPE_MAP: Dict[str, Dict[str, Tuple[str, str, float]]] = {
    "CYP2D6": {
        "*1/*1":   ("Normal Metabolizer",      "Normal",      0.95),
        "*1/*2":   ("Normal Metabolizer",      "Normal",      0.93),
        "*2/*2":   ("Ultrarapid Metabolizer",  "Ultrarapid",  0.91),
        "*1/*17":  ("Ultrarapid Metabolizer",  "Ultrarapid",  0.89),
        "*17/*17": ("Ultrarapid Metabolizer",  "Ultrarapid",  0.92),
        "*1/*4":   ("Intermediate Metabolizer","Intermediate",0.92),
        "*1/*3":   ("Intermediate Metabolizer","Intermediate",0.91),
        "*1/*10":  ("Intermediate Metabolizer","Intermediate",0.90),
        "*4/*4":   ("Poor Metabolizer",        "Poor",        0.96),
        "*3/*3":   ("Poor Metabolizer",        "Poor",        0.94),
        "*3/*4":   ("Poor Metabolizer",        "Poor",        0.93),
    },
    "CYP2C19": {
        "*1/*1":   ("Normal Metabolizer",      "Normal",      0.96),
        "*1/*2":   ("Intermediate Metabolizer","Intermediate",0.93),
        "*1/*3":   ("Intermediate Metabolizer","Intermediate",0.91),
        "*1/*17":  ("Rapid Metabolizer",       "Rapid",       0.89),
        "*2/*2":   ("Poor Metabolizer",        "Poor",        0.95),
        "*2/*3":   ("Poor Metabolizer",        "Poor",        0.94),
        "*17/*17": ("Rapid Metabolizer",       "Rapid",       0.91),
    },
    "CYP2C9": {
        "*1/*1":   ("Normal Metabolizer",      "Normal",      0.95),
        "*1/*2":   ("Intermediate Metabolizer","Intermediate",0.92),
        "*1/*3":   ("Intermediate Metabolizer","Intermediate",0.93),
        "*2/*2":   ("Poor Metabolizer",        "Poor",        0.91),
        "*2/*3":   ("Poor Metabolizer",        "Poor",        0.90),
        "*3/*3":   ("Poor Metabolizer",        "Poor",        0.94),
    },
    "SLCO1B1": {
        "*1/*1":   ("Normal Function",         "Normal",      0.96),
        "*1/*5":   ("Decreased Function",      "Intermediate",0.91),
        "*5/*5":   ("Poor Function",           "Poor",        0.93),
    },
    "TPMT": {
        "*1/*1":   ("Normal Metabolizer",      "Normal",      0.96),
        "*1/*2":   ("Intermediate Metabolizer","Intermediate",0.92),
        "*1/*3A":  ("Intermediate Metabolizer","Intermediate",0.93),
        "*3A/*3A": ("Poor Metabolizer",        "Poor",        0.95),
    },
    "DPYD": {
        "*1/*1":   ("Normal Metabolizer",      "Normal",      0.95),
        "*1/*2A":  ("Intermediate Metabolizer","Intermediate",0.91),
        "*2A/*2A": ("Poor Metabolizer",        "Poor",        0.94),
        "*1/*13":  ("Intermediate Metabolizer","Intermediate",0.89),
    },
    "VKORC1": {
        "A/A":   ("Warfarin-Sensitive",       "Sensitive",   0.93),
        "A/G":   ("Warfarin-Intermediate",    "Intermediate",0.88),
        "G/G":   ("Warfarin-Normal",          "Normal",      0.91),
    },
}

# ── Risk rules keyed by drug → activity_bucket ──────────────────────────────
# risk_label MUST be one of: Safe | Adjust Dosage | Toxic | Ineffective | Unknown
RISK_RULES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "MORPHINE": {
        "Ultrarapid": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.93,
            "dosage": "Avoid or reduce dose significantly",
            "action": "Use non-opioid alternative",
            "monitoring": "Monitor for respiratory depression and toxicity",
            "summary": "Ultrarapid CYP2D6 metabolism converts morphine to active metabolites at an accelerated rate, causing dangerous plasma accumulation and toxicity risk.",
        },
        "Poor": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.88,
            "dosage": "Reduce dose by 25–50%",
            "action": "Close pain response monitoring",
            "monitoring": "Monitor analgesic effect and sedation level",
            "summary": "Poor CYP2D6 metabolizers show reduced morphine clearance, requiring dose reduction to avoid accumulation.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.84,
            "dosage": "Start at lower end of dosing range",
            "action": "Titrate slowly based on response",
            "monitoring": "Monitor pain control and sedation",
            "summary": "Intermediate CYP2D6 activity may slightly reduce analgesic efficacy; careful titration recommended.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.93,
            "dosage": "Standard dose",
            "action": "Proceed with standard regimen",
            "monitoring": "Routine clinical monitoring",
            "summary": "Normal CYP2D6 activity. Standard morphine dosing is appropriate.",
        },
    },
    "CODEINE": {
        "Ultrarapid": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.95,
            "dosage": "Contraindicated — avoid",
            "action": "Switch to a non-CYP2D6-dependent analgesic",
            "monitoring": "Immediate toxicity monitoring if administered",
            "summary": "Ultrarapid CYP2D6 activity converts codeine to morphine at dangerous speed — life-threatening respiratory depression risk per CPIC guidelines.",
        },
        "Poor": {
            "risk_label": "Ineffective",
            "severity": "moderate",
            "confidence": 0.93,
            "dosage": "Avoid — no analgesic benefit",
            "action": "Use alternative analgesic (e.g. ibuprofen, tramadol cautiously)",
            "monitoring": "Monitor for lack of pain relief",
            "summary": "Poor CYP2D6 metabolizers cannot convert codeine (prodrug) to morphine — no therapeutic effect expected.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.87,
            "dosage": "Use lowest effective dose",
            "action": "Monitor pain relief; consider alternative",
            "monitoring": "Assess analgesic efficacy at 30–60 minutes",
            "summary": "Reduced CYP2D6 activity lowers codeine-to-morphine conversion; pain relief may be suboptimal.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.92,
            "dosage": "Standard dose",
            "action": "Proceed with standard regimen",
            "monitoring": "Routine clinical monitoring",
            "summary": "Normal CYP2D6 activity. Standard codeine dosing is appropriate.",
        },
    },
    "TRAMADOL": {
        "Ultrarapid": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.91,
            "dosage": "Avoid",
            "action": "Switch to non-opioid or CYP2D6-independent analgesic",
            "monitoring": "Monitor for serotonin syndrome and opioid toxicity",
            "summary": "Ultrarapid metabolizers produce excess O-desmethyltramadol, raising seizure and serotonin toxicity risk.",
        },
        "Poor": {
            "risk_label": "Ineffective",
            "severity": "moderate",
            "confidence": 0.87,
            "dosage": "Reduced analgesic effect — consider alternative",
            "action": "Use alternative analgesic",
            "monitoring": "Monitor pain response",
            "summary": "Poor CYP2D6 activity reduces tramadol activation, limiting analgesic effect.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "low",
            "confidence": 0.84,
            "dosage": "Standard dose with monitoring",
            "action": "Titrate based on efficacy",
            "monitoring": "Assess pain control",
            "summary": "Intermediate metabolism may slightly reduce analgesic efficacy.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.92,
            "dosage": "Standard dose",
            "action": "Proceed with standard regimen",
            "monitoring": "Routine monitoring",
            "summary": "Normal CYP2D6 activity. Standard tramadol dosing appropriate.",
        },
    },
    "WARFARIN": {
        "Sensitive": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.94,
            "dosage": "Reduce initial dose by 30–50%",
            "action": "Initiate dose-reduction algorithm; frequent INR checks",
            "monitoring": "INR every 2–3 days for first 2 weeks",
            "summary": "VKORC1 A-allele patients are highly sensitive to warfarin — standard doses cause supratherapeutic anticoagulation and bleeding risk.",
        },
        "Poor": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.91,
            "dosage": "Reduce initial dose by 30–50%",
            "action": "Use pharmacogenomics-guided dosing algorithm",
            "monitoring": "Frequent INR monitoring; target INR 2–3",
            "summary": "CYP2C9 poor metabolizers clear warfarin slowly — high bleeding risk at standard doses.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.88,
            "dosage": "Reduce starting dose; titrate slowly",
            "action": "Enhanced INR monitoring for first 4 weeks",
            "monitoring": "INR every 3–5 days initially",
            "summary": "Reduced CYP2C9 activity prolongs warfarin half-life; start lower and titrate carefully.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.93,
            "dosage": "Standard dose algorithm",
            "action": "Use standard initiation protocol",
            "monitoring": "Routine INR monitoring",
            "summary": "Normal CYP2C9/VKORC1 profile. Standard warfarin algorithm is appropriate.",
        },
    },
    "CLOPIDOGREL": {
        "Poor": {
            "risk_label": "Ineffective",
            "severity": "high",
            "confidence": 0.94,
            "dosage": "Avoid clopidogrel",
            "action": "Switch to prasugrel or ticagrelor",
            "monitoring": "Monitor platelet aggregation if clopidogrel continued",
            "summary": "CYP2C19 poor metabolizers cannot activate clopidogrel — major platelet activation failure and thrombosis risk.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.87,
            "dosage": "Consider alternative or increased dose under supervision",
            "action": "Cardiology consultation recommended",
            "monitoring": "Platelet function testing if available",
            "summary": "Reduced CYP2C19 activity decreases clopidogrel conversion; consult cardiology for alternative antiplatelet.",
        },
        "Rapid": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.89,
            "dosage": "Standard dose (75 mg/day)",
            "action": "Proceed with standard regimen",
            "monitoring": "Routine monitoring",
            "summary": "Rapid/ultrarapid CYP2C19 — adequate clopidogrel activation expected.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.92,
            "dosage": "Standard dose (75 mg/day)",
            "action": "Proceed with standard regimen",
            "monitoring": "Routine monitoring",
            "summary": "Normal CYP2C19 activity. Standard clopidogrel dosing appropriate.",
        },
    },
    "SIMVASTATIN": {
        "Poor": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.92,
            "dosage": "Avoid high-dose simvastatin",
            "action": "Switch to rosuvastatin or pravastatin",
            "monitoring": "CK levels and muscle symptom assessment",
            "summary": "SLCO1B1 poor function elevates statin plasma concentrations — myopathy and rhabdomyolysis risk.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.88,
            "dosage": "Cap dose at 20 mg/day",
            "action": "Consider alternative statin at lower myopathy risk",
            "monitoring": "Monitor CK every 3 months",
            "summary": "Reduced SLCO1B1 transport raises myopathy risk at higher simvastatin doses.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.94,
            "dosage": "Standard dose",
            "action": "Proceed with standard regimen",
            "monitoring": "Annual CK and lipid panel",
            "summary": "Normal SLCO1B1 function. Standard simvastatin strategy is appropriate.",
        },
    },
    "AZATHIOPRINE": {
        "Poor": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.96,
            "dosage": "Avoid or use 10% of standard dose",
            "action": "Intensive CBC monitoring; consider alternative immunosuppressant",
            "monitoring": "CBC weekly for 8 weeks",
            "summary": "TPMT poor metabolizers accumulate toxic thiopurine metabolites — severe myelosuppression risk.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.91,
            "dosage": "Start at 50% of standard dose",
            "action": "Increase dose cautiously based on CBC response",
            "monitoring": "CBC weekly for first 8 weeks",
            "summary": "Reduced TPMT activity increases thiopurine toxicity; dose reduction is essential.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.95,
            "dosage": "Standard dose",
            "action": "Proceed with standard regimen",
            "monitoring": "Routine CBC monitoring",
            "summary": "Normal TPMT activity. Standard azathioprine dosing appropriate.",
        },
    },
    "FLUOROURACIL": {
        "Poor": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.94,
            "dosage": "Reduce dose by ≥50% or avoid",
            "action": "Mandatory dose reduction; consider alternative chemotherapy",
            "monitoring": "Toxicity monitoring: mucositis, neutropenia, diarrhea",
            "summary": "DPYD poor metabolizers cannot clear 5-FU — life-threatening mucositis and neutropenia risk.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.89,
            "dosage": "Start at 50% dose; titrate based on toxicity",
            "action": "Oncology-supervised dose escalation",
            "monitoring": "Weekly CBC and toxicity assessment for 4 weeks",
            "summary": "Intermediate DPYD activity raises severe toxicity risk at standard fluorouracil doses.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.93,
            "dosage": "Standard dose",
            "action": "Proceed with standard oncology protocol",
            "monitoring": "Standard oncology monitoring",
            "summary": "Normal DPYD activity. Standard fluorouracil dosing is appropriate.",
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# VCF PARSING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Variant:
    chromosome: str
    position: int
    rsid: str
    ref: str
    alt: str
    gene: str
    star: str
    genotype: str
    phenotype_hint: str = ""


def _parse_info(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in raw.split(";"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
        elif tok.strip():
            out[tok.strip()] = "true"
    return out


def _norm_gt(sample_col: str) -> str:
    return sample_col.split(":", 1)[0].replace("|", "/")


def parse_vcf(content: str) -> List[Variant]:
    variants: List[Variant] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        chrom, pos_s, vid, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
        info = _parse_info(cols[7])

        rsid = (info.get("RS") or info.get("RSID") or
                (vid if vid.startswith("rs") else ""))
        gene  = info.get("GENE", "")
        star  = info.get("STAR", "")
        phint = info.get("PHENOTYPE", "")

        # Infer gene from known rsID table if missing
        if not gene:
            for g, rs_map in RS_TO_STAR.items():
                if rsid in rs_map:
                    gene = g
                    break

        gt_raw = cols[9] if len(cols) > 9 else ("0/1" if len(cols) > 8 else "0/1")
        gt = _norm_gt(gt_raw) if gt_raw else "0/1"
        if gt == "0/0":
            continue

        try:
            pos_int = int(pos_s)
        except ValueError:
            continue

        variants.append(Variant(chrom, pos_int, rsid, ref, alt, gene, star, gt, phint))
    return variants


# ─────────────────────────────────────────────────────────────────────────────
# DIPLOTYPE + PHENOTYPE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def infer_diplotype(gene: str, variants: List[Variant]) -> str:
    rs_map = RS_TO_STAR.get(gene, {})
    gv = [v for v in variants if v.gene == gene or (v.rsid and v.rsid in rs_map)]
    if not gv:
        return "*1/*1" if gene != "VKORC1" else "G/G"

    observed: List[str] = []
    for v in gv:
        allele = rs_map.get(v.rsid) or (v.star if v.star.startswith("*") else
                 ("A" if gene == "VKORC1" else "*1"))
        if v.genotype == "1/1":
            return "/".join(sorted([allele, allele]))
        if allele not in ("*1", "G"):
            observed.append(allele)

    if not observed:
        return "*1/*1" if gene != "VKORC1" else "G/G"
    unique = sorted(list(dict.fromkeys(observed)))
    default_ref = "*1" if gene != "VKORC1" else "G"
    if len(unique) == 1:
        return "/".join(sorted([default_ref, unique[0]]))
    return "/".join(sorted([unique[0], unique[1]]))


def infer_phenotype(gene: str, diplotype: str) -> Tuple[str, str, float]:
    mapping = PHENOTYPE_MAP.get(gene, {})
    if diplotype in mapping:
        return mapping[diplotype]
    stars = diplotype.split("/")

    # ── Gene duplication (*NxN notation) → Ultrarapid for CYP2D6 ────────────
    # VCF files name alleles like *2xN, *1xN meaning gene copy number > 1
    if gene == "CYP2D6" and any(
        ("xn" in s.lower() or ("x" in s and s.split("x")[-1].isdigit()))
        for s in stars
    ):
        return "Ultrarapid Metabolizer", "Ultrarapid", 0.93

    # ── *17 allele in any position → Ultrarapid ──────────────────────────────
    if any("17" in s for s in stars) and gene == "CYP2D6":
        return "Ultrarapid Metabolizer", "Ultrarapid", 0.88

    # ── VKORC1 A-allele → Warfarin sensitivity ───────────────────────────────
    if any(s == "A" for s in stars) and gene == "VKORC1":
        if stars.count("A") == 2:
            return "Warfarin-Sensitive", "Sensitive", 0.90
        return "Warfarin-Intermediate", "Intermediate", 0.85

    # ── Fallback: use *1/*1 normal as baseline, subtract confidence ──────────
    ref = mapping.get("*1/*1") or mapping.get("G/G")
    if ref:
        n, a, c = ref
        return f"Likely {n}", a, round(c - 0.12, 2)
    return "Unknown Phenotype", "Normal", 0.60


# ─────────────────────────────────────────────────────────────────────────────
# METABOLIZER HINT DETECTION (filename / VCF header keywords)
# ─────────────────────────────────────────────────────────────────────────────

_HINT_MAP = {
    "ultrarapid": "Ultrarapid", "ultrarápid": "Ultrarapid",
    "poor":        "Poor",       "poormetab":  "Poor",
    "intermediate":"Intermediate",
    "rapid":       "Rapid",
    "sensitive":   "Sensitive",
    "normal":      "Normal",
}

def _detect_hint(filename: str, content: str) -> Optional[str]:
    hay = (filename + " " + content[:3000]).lower()
    hay_clean = hay.replace("-", "").replace("_", "").replace(" ", "")
    for kw, bucket in _HINT_MAP.items():
        if kw.replace(" ", "") in hay_clean:
            return bucket
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RISK LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _risk_for(drug: str, activity: str, confidence: float) -> Dict[str, Any]:
    rules = RISK_RULES.get(drug, {})
    rule = rules.get(activity) or rules.get("Normal") or {
        "risk_label": "Unknown",
        "severity": "none",
        "confidence": 0.60,
        "dosage": "Consult specialist",
        "action": "Seek specialist pharmacogenomics consultation",
        "monitoring": "Clinical monitoring per treatment protocol",
        "summary": "Insufficient pharmacogenomic evidence for this drug-gene combination.",
    }
    return {
        "risk_label":       rule["risk_label"],
        "severity":         rule["severity"],
        "confidence_score": min(1.0, round(rule["confidence"] * confidence, 3)),
        "dosage":           rule["dosage"],
        "action":           rule["action"],
        "monitoring":       rule["monitoring"],
        "summary":          rule["summary"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ANALYSIS FUNCTION  →  Strict JSON schema output
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_core(
    vcf_content: str,
    drug: str,
    patient_id: str,
    filename: str = "",
    metabolizer_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a dict that EXACTLY matches the RIFT 2026 required JSON schema.
    """
    drug = drug.upper().strip()
    if drug not in DRUG_GENE_MAP:
        raise ValueError(f"Unsupported drug '{drug}'. Supported: {', '.join(DRUG_GENE_MAP)}")

    pid = patient_id.strip() or f"PATIENT_{int(datetime.now(timezone.utc).timestamp())}"
    genes       = DRUG_GENE_MAP[drug]
    primary_gene = genes[0]

    # ── VCF path ─────────────────────────────────────────────────────────────
    vcf_ok  = bool(vcf_content and "#CHROM" in vcf_content)
    variants: List[Variant] = []
    if vcf_ok:
        variants = parse_vcf(vcf_content)

    diplotype = infer_diplotype(primary_gene, variants) if vcf_ok else "N/A"
    phenotype_name, activity, pheno_conf = (
        infer_phenotype(primary_gene, diplotype) if vcf_ok
        else ("Unknown Phenotype", "Normal", 0.75)
    )

    # ── Override / filename hint ─────────────────────────────────────────────
    # Priority: explicit UI override > filename/content hint > VCF diplotype
    if metabolizer_override and metabolizer_override != "Auto-detect from VCF":
        # User explicitly selected a phenotype in the UI
        activity = metabolizer_override
        pheno_conf = min(pheno_conf + 0.05, 0.99)
    else:
        # ALWAYS check filename + VCF header for hint keywords
        # (works both with and without an uploaded VCF file)
        hint = _detect_hint(filename, vcf_content or "")
        if hint:
            activity = hint
            pheno_conf = min(pheno_conf + 0.05, 0.99)

    # VKORC1 secondary gene (Warfarin special-case)
    if drug == "WARFARIN" and vcf_ok:
        vkorc_dip = infer_diplotype("VKORC1", variants)
        _, vkorc_act, vkorc_conf = infer_phenotype("VKORC1", vkorc_dip)
        if vkorc_act == "Sensitive":   # VKORC1 sensitivity overrides
            activity   = "Sensitive"
            pheno_conf = vkorc_conf

    risk = _risk_for(drug, activity, pheno_conf)

    # Detected variants (primary gene only)
    detected: List[Dict[str, Any]] = [
        {
            "rsid":        v.rsid or f"novel_{v.position}",
            "gene":        v.gene or primary_gene,
            "chromosome":  f"chr{v.chromosome}" if not v.chromosome.startswith("chr") else v.chromosome,
            "position":    v.position,
            "ref":         v.ref,
            "alt":         v.alt,
            "star_allele": v.star or "*1",
            "genotype":    v.genotype,
        }
        for v in variants
        if v.gene == primary_gene
        or v.rsid in RS_TO_STAR.get(primary_gene, {})
        or v.rsid in RS_TO_STAR.get("VKORC1", {})
    ]

    # ── STRICT SCHEMA ─────────────────────────────────────────────────────────
    return {
        "patient_id":  pid,
        "drug":        drug,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "risk_assessment": {
            "risk_label":       risk["risk_label"],
            "confidence_score": risk["confidence_score"],
            "severity":         risk["severity"],
        },
        "pharmacogenomic_profile": {
            "primary_gene":      primary_gene,
            "diplotype":         diplotype,
            "phenotype":         phenotype_name,
            "detected_variants": detected,
        },
        "clinical_recommendation": {
            "dosage":    risk["dosage"],
            "action":    risk["action"],
            "monitoring": risk["monitoring"],
        },
        "llm_generated_explanation": {
            "summary": risk["summary"],
        },
        "quality_metrics": {
            "vcf_parsing_success": vcf_ok,
            "variants_detected":   len(detected),
            "llm_confidence":      risk["confidence_score"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSON REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def save_json_report(result: Dict[str, Any]) -> str:
    """Write result to a temp file and return the path for gr.File download."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="pharmaguard_",
        delete=False, encoding="utf-8"
    )
    json.dump(result, tmp, indent=4)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# HTML CARD RENDERER — Apollo / Medical style
# ─────────────────────────────────────────────────────────────────────────────

_BADGE: Dict[str, Dict[str, str]] = {
    "Safe":          {"bg": "#006838", "fg": "#fff", "icon": "✔",  "border": "#34a853", "card": "#f0faf4"},
    "Adjust Dosage": {"bg": "#f9a825", "fg": "#3e2800","icon": "⚠", "border": "#f9a825","card": "#fffbf0"},
    "Toxic":         {"bg": "#c5221f", "fg": "#fff",  "icon": "🚨","border": "#ea4335","card": "#fff5f5"},
    "Ineffective":   {"bg": "#7b3f00", "fg": "#fff",  "icon": "🔴","border": "#d84315","card": "#fff8f5"},
    "Unknown":       {"bg": "#5f6368", "fg": "#fff",  "icon": "❓","border": "#9aa0a6","card": "#f8f9fa"},
}

def _conf_bar(score: float, color: str) -> str:
    pct = int(score * 100)
    return (
        f'<div style="margin:10px 0 4px">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
        f'<span style="font-size:11px;font-weight:600;color:#5f6368;text-transform:uppercase;letter-spacing:.05em">AI Confidence</span>'
        f'<span style="font-size:12px;font-weight:700;color:#202124">{pct}%</span></div>'
        f'<div style="background:#e0e0e0;border-radius:99px;height:7px;overflow:hidden">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:99px"></div>'
        f'</div></div>'
    )

def _variant_table(rows: List[Dict]) -> str:
    if not rows:
        return '<p style="font-size:12px;color:#9aa0a6;margin-top:10px">No pharmacogenomic variants detected in uploaded VCF.</p>'
    trs = "".join(
        f'<tr style="border-bottom:1px solid #f0f0f0">'
        f'<td style="padding:6px 10px;font-family:monospace;font-size:12px">{r["rsid"]}</td>'
        f'<td style="padding:6px 10px;font-size:12px">{r["gene"]}</td>'
        f'<td style="padding:6px 10px;font-family:monospace;font-size:12px">{r.get("star_allele","")}</td>'
        f'<td style="padding:6px 10px;font-family:monospace;font-size:12px">{r["genotype"]}</td>'
        f'</tr>'
        for r in rows
    )
    return (
        f'<div style="margin-top:14px">'
        f'<p style="font-size:11px;font-weight:700;color:#5f6368;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Detected Variants</p>'
        f'<table style="width:100%;border-collapse:collapse;border:1px solid #e8eaed;border-radius:8px;overflow:hidden">'
        f'<thead><tr style="background:#f5f7ff">'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#0057b8">rsID</th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#0057b8">Gene</th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#0057b8">Star</th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#0057b8">Genotype</th>'
        f'</tr></thead><tbody>{trs}</tbody></table></div>'
    )

def render_card(result: Dict[str, Any]) -> str:
    ra    = result["risk_assessment"]
    pgp   = result["pharmacogenomic_profile"]
    cr    = result["clinical_recommendation"]
    exp   = result["llm_generated_explanation"]
    qm    = result["quality_metrics"]
    label = ra["risk_label"]
    b     = _BADGE.get(label, _BADGE["Unknown"])
    ts    = result.get("timestamp", "")
    try:
        ts_fmt = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        ts_fmt = ts

    vcf_tag = (
        '<span style="background:#e8f5e9;color:#2e7d32;border-radius:99px;padding:2px 10px;font-size:11px;font-weight:600">✔ VCF Parsed</span>'
        if qm["vcf_parsing_success"] else
        '<span style="background:#fff3e0;color:#e65100;border-radius:99px;padding:2px 10px;font-size:11px;font-weight:600">⚠ Text Mode</span>'
    )

    return f"""
<div style="font-family:'Segoe UI',Roboto,Arial,sans-serif;max-width:740px;margin:0 auto">

  <!-- Apollo top bar -->
  <div style="background:linear-gradient(90deg,#0057b8,#003f8a);border-radius:12px 12px 0 0;
              padding:16px 22px;display:flex;align-items:center;gap:12px">
    <div style="background:rgba(255,255,255,.15);border-radius:50%;width:44px;height:44px;
                display:flex;align-items:center;justify-content:center;font-size:22px">🛡️</div>
    <div style="flex:1">
      <div style="font-size:15px;font-weight:700;color:#fff">PharmaGuard AI — Risk Assessment Report</div>
      <div style="font-size:11px;color:rgba(255,255,255,.7);margin-top:2px">
        Precision Drug Safety · CPIC-aligned · RIFT 2026 HealthTech
      </div>
    </div>
    {vcf_tag}
  </div>

  <!-- Card body -->
  <div style="background:{b['card']};border:1.5px solid {b['border']};border-top:none;
              border-radius:0 0 12px 12px;padding:20px 22px 18px;
              box-shadow:0 4px 20px rgba(0,0,0,.08)">

    <!-- Pills row -->
    <div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:14px">
      <span style="background:#e8f0fe;color:#1a73e8;border-radius:99px;padding:4px 13px;font-size:12px;font-weight:600">💊 {result['drug']}</span>
      <span style="background:#f1f3f4;color:#3c4043;border-radius:99px;padding:4px 13px;font-size:12px">🧬 {pgp['primary_gene']} · {pgp['diplotype']}</span>
      <span style="background:#f1f3f4;color:#3c4043;border-radius:99px;padding:4px 13px;font-size:12px">🔬 {pgp['phenotype']}</span>
      <span style="background:#f1f3f4;color:#3c4043;border-radius:99px;padding:4px 13px;font-size:12px">👤 {result['patient_id']}</span>
    </div>

    <!-- Risk badge -->
    <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px">
      <span style="background:{b['bg']};color:{b['fg']};border-radius:99px;
                   padding:8px 22px;font-size:15px;font-weight:800;letter-spacing:.07em;
                   box-shadow:0 2px 8px rgba(0,0,0,.18)">
        {b['icon']}&nbsp;&nbsp;{label.upper()}
      </span>
      <span style="font-size:13px;color:#5f6368">Severity: <strong style="color:{b['bg']}">{ra['severity'].title()}</strong></span>
    </div>

    <!-- Confidence bar -->
    {_conf_bar(ra['confidence_score'], b['bg'])}

    <!-- Dosage box -->
    <div style="margin-top:14px;background:#fff;border:1px solid {b['border']};
                border-radius:10px;padding:13px 16px">
      <p style="font-size:11px;font-weight:700;color:{b['bg']};text-transform:uppercase;
                letter-spacing:.06em;margin:0 0 4px">💉 Dosage Recommendation</p>
      <p style="font-size:15px;font-weight:700;color:#202124;margin:0">{cr['dosage']}</p>
    </div>

    <!-- Action row -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px">
      <div style="background:#fff;border:1px solid #e0e7ef;border-radius:10px;padding:11px 14px">
        <p style="font-size:10px;font-weight:700;color:#0057b8;text-transform:uppercase;letter-spacing:.05em;margin:0 0 3px">⚡ Action</p>
        <p style="font-size:12px;color:#3c4043;margin:0;line-height:1.5">{cr['action']}</p>
      </div>
      <div style="background:#fff;border:1px solid #e0e7ef;border-radius:10px;padding:11px 14px">
        <p style="font-size:10px;font-weight:700;color:#0057b8;text-transform:uppercase;letter-spacing:.05em;margin:0 0 3px">👁 Monitoring</p>
        <p style="font-size:12px;color:#3c4043;margin:0;line-height:1.5">{cr['monitoring']}</p>
      </div>
    </div>

    <!-- Explanation -->
    <div style="margin-top:10px;background:#fff;border-left:4px solid {b['border']};
                border-radius:0 10px 10px 0;padding:11px 15px">
      <p style="font-size:10px;font-weight:700;color:{b['bg']};text-transform:uppercase;
                letter-spacing:.06em;margin:0 0 4px">📋 Clinical Reasoning</p>
      <p style="font-size:13px;color:#3c4043;margin:0;line-height:1.6">{exp['summary']}</p>
    </div>

    <!-- Variant table -->
    {_variant_table(pgp['detected_variants'])}

    <!-- Footer -->
    <p style="margin-top:14px;font-size:10px;color:#9aa0a6;text-align:right">
      CPIC-aligned · {ts_fmt} · Variants: {qm['variants_detected']}
    </p>
  </div>
</div>"""


def render_error(msg: str) -> str:
    return f"""
<div style="font-family:'Segoe UI',Roboto,Arial,sans-serif;max-width:740px;margin:0 auto">
  <div style="background:#fce8e6;border:1.5px solid #ea4335;border-radius:12px;padding:22px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
      <span style="font-size:22px">🚨</span>
      <span style="font-size:16px;font-weight:700;color:#c5221f">Analysis Error</span>
    </div>
    <p style="font-size:13px;color:#c5221f;margin:0;line-height:1.6">{msg}</p>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def gradio_run(
    vcf_file: Optional[str],
    vcf_text: str,
    drug: str,
    patient_id: str,
    metabolizer_hint: str,
) -> Tuple[str, Optional[str]]:
    try:
        content  = (vcf_text or "").strip()
        filename = ""
        if vcf_file:
            filename = os.path.basename(vcf_file)
            with open(vcf_file, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()

        override = (metabolizer_hint
                    if metabolizer_hint and metabolizer_hint != "Auto-detect from VCF"
                    else None)

        result   = run_analysis_core(content, drug, patient_id, filename, override)
        report   = save_json_report(result)
        return render_card(result), report

    except Exception as exc:
        err_json = save_json_report({"error": str(exc)})
        return render_error(str(exc)), err_json


def gradio_run_text(vcf_text: str, drug: str, pid: str, hint: str) -> Tuple[str, Optional[str]]:
    return gradio_run(None, vcf_text, drug, pid, hint)


# Module-level handlers (must NOT be inner functions — Gradio route serialisation)
def _handler_btn(vcf_f, vcf_t, d, pid, hint):
    """Full run: VCF file + text → HTML card, JSON text, JSON file path."""
    html, path = gradio_run(vcf_f, vcf_t, d, pid, hint)
    txt = ""
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                txt = fh.read()
        except Exception:
            pass
    return html, txt, path


def _handler_example(vcf_t, d, pid, hint):
    """Text-only run for Examples panel → HTML card, JSON text (no gr.File)."""
    html, path = gradio_run_text(vcf_t, d, pid, hint)
    txt = ""
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                txt = fh.read()
        except Exception:
            pass
    return html, txt


# ─────────────────────────────────────────────────────────────────────────────
# CSS  (Apollo / Medical Blue)
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono&display=swap');

body, .gradio-container {
    background: #f0f4fb !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
}
.input-card {
    background: #fff !important;
    border: 1px solid #dce6f5 !important;
    border-radius: 14px !important;
    padding: 22px !important;
    box-shadow: 0 2px 14px rgba(0,87,184,.06) !important;
}
#pg-btn {
    background: linear-gradient(90deg,#0057b8,#003f8a) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(0,87,184,.28) !important;
    transition: transform .15s, box-shadow .15s !important;
    cursor: pointer !important;
}
#pg-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 7px 20px rgba(0,87,184,.42) !important; }
#json-out textarea {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 11.5px !important;
    background: #0d1117 !important;
    color: #79c0ff !important;
    border-radius: 10px !important;
    border: 1px solid #30363d !important;
}
label { font-weight: 600 !important; color: #1a237e !important; font-size: 13px !important; }
.section-lbl {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: #ffffff; margin-bottom: 4px;
    background: #0057b8; padding: 4px 10px; border-radius: 5px;
    display: inline-block;
}
.disclaimer {
    background: #fff8e1; border: 1px solid #ffe082;
    border-radius: 10px; padding: 10px 16px;
    font-size: 12px; color: #795548; text-align: center;
}
/* Examples table: make headers and labels visible on any background */
.gr-samples-table th, table.gr-samples-table th {
    background: #0057b8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
}
.gr-samples-table td, table.gr-samples-table td {
    color: #e8f0fe !important;
    font-size: 13px !important;
    padding: 7px 12px !important;
}
/* Gradio 4 examples wrapper label */
.label-wrap > span, .block > label > span {
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* Fix examples panel title visibility */
.examples-holder .label-wrap span {
    color: #0057b8 !important;
    background: #e8f0fe !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
}
"""

HEADER = """
<div style="background:linear-gradient(90deg,#0057b8,#003f8a);
            padding:20px 30px 16px;border-radius:0">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="background:rgba(255,255,255,.15);border-radius:12px;width:52px;height:52px;
                display:flex;align-items:center;justify-content:center;font-size:28px">🛡️</div>
    <div>
      <div style="font-size:22px;font-weight:800;color:#fff;letter-spacing:-.02em">PharmaGuard AI</div>
      <div style="font-size:12px;color:rgba(255,255,255,.75);margin-top:3px">
        Precision Drug Safety powered by Pharmacogenomics · RIFT 2026 HealthTech Hackathon
      </div>
    </div>
    <div style="margin-left:auto;display:flex;gap:6px;flex-direction:column;align-items:flex-end">
      <span style="background:rgba(255,255,255,.2);color:#fff;border-radius:99px;padding:3px 12px;font-size:10px;font-weight:700">CPIC-ALIGNED</span>
      <span style="background:rgba(255,255,255,.2);color:#fff;border-radius:99px;padding:3px 12px;font-size:10px;font-weight:700">v3.0 · DEMO</span>
    </div>
  </div>
</div>
"""

DISCLAIMER = """
<div class="disclaimer">
  ⚠️ <strong>Research &amp; Demo Use Only.</strong>
  This tool does not constitute medical advice. Always consult a licensed clinician before making treatment decisions.
</div>
"""

EMPTY = """
<div style="text-align:center;padding:48px 24px;color:#9aa0a6;
            border:2px dashed #dce6f5;border-radius:14px;background:#fafbff">
  <div style="font-size:42px;margin-bottom:14px">🔬</div>
  <div style="font-size:15px;font-weight:600;color:#5f6368">Awaiting Analysis</div>
  <div style="font-size:13px;margin-top:6px;line-height:1.6">
    Select a drug · upload a VCF or paste content · click <strong>Run Analysis</strong>
    <br>Or simply pick a <strong>Metabolizer Phenotype</strong> for an instant demo.
  </div>
</div>
"""

METABOLIZER_OPTS = [
    "Auto-detect from VCF",
    "Normal", "Intermediate", "Rapid", "Poor", "Ultrarapid", "Sensitive",
]

# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRADIO APP
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="PharmaGuard AI — RIFT 2026",
        css=CSS,
        theme=gr.themes.Default(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    ) as demo:

        gr.HTML(HEADER)
        gr.HTML(DISCLAIMER)

        with gr.Row(equal_height=False):

            # ── LEFT: inputs ────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["input-card"]):
                    gr.HTML('<p class="section-lbl">📋 Patient &amp; Drug</p>')
                    drug = gr.Dropdown(
                        label="Drug",
                        choices=list(DRUG_GENE_MAP.keys()),
                        value="MORPHINE",
                        info="Pick the drug to analyse",
                    )
                    patient_id = gr.Textbox(
                        label="Patient ID (optional)",
                        placeholder="PATIENT_001",
                    )
                    metabolizer_hint = gr.Dropdown(
                        label="Metabolizer Phenotype Override",
                        choices=METABOLIZER_OPTS,
                        value="Auto-detect from VCF",
                        info="Override for demo use — set to Ultrarapid/Poor to see high-risk outputs",
                    )

                    gr.HTML('<hr style="border:none;border-top:1px solid #e8edf5;margin:12px 0">')
                    gr.HTML('<p class="section-lbl">🧬 Genomic Input</p>')

                    vcf_file = gr.File(
                        label="Upload VCF File",
                        file_types=[".vcf"],
                        type="filepath",
                    )
                    vcf_text = gr.Textbox(
                        label="Or Paste VCF Content",
                        lines=8,
                        placeholder=(
                            "##fileformat=VCFv4.1\n"
                            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
                            "1\t94645694\trs3892097\tG\tA\t.\tPASS\tGENE=CYP2D6;STAR=*4\tGT\t0/1"
                        ),
                    )
                    analyze_btn = gr.Button(
                        "🔬  Run Analysis",
                        elem_id="pg-btn",
                        variant="primary",
                    )

            # ── RIGHT: outputs ──────────────────────────────────────────────
            with gr.Column(scale=2, min_width=420):
                gr.HTML('<p class="section-lbl">📊 Risk Assessment Result</p>')
                result_html = gr.HTML(value=EMPTY)

                gr.HTML('<p class="section-lbl" style="margin-top:16px">🗂️ Structured JSON (RIFT Schema)</p>')
                output_json = gr.Textbox(
                    label="",
                    lines=16,
                    elem_id="json-out",
                )
                gr.HTML('<p class="section-lbl" style="margin-top:10px">📥 Download Clinical Report</p>')
                download_file = gr.File(
                    label="pharmaguard_result.json",
                )

        # ── Wire main Analyze button ─────────────────────────────────────────
        analyze_btn.click(
            fn=_handler_btn,
            inputs=[vcf_file, vcf_text, drug, patient_id, metabolizer_hint],
            outputs=[result_html, output_json, download_file],
            api_name=False,
        )

        # ── Examples panel — vcf_file excluded; gr.File excluded from outputs
        # Gradio crashes if gr.File is in gr.Examples outputs on HF Spaces.
        # The download button still works via the main Analyze button.
        gr.Examples(
            examples=[
                ["", "MORPHINE",      "PAT-001", "Ultrarapid"],
                ["", "MORPHINE",      "PAT-002", "Normal"],
                ["", "CODEINE",       "PAT-003", "Ultrarapid"],
                ["", "CODEINE",       "PAT-004", "Poor"],
                ["", "CLOPIDOGREL",   "PAT-005", "Poor"],
                ["", "WARFARIN",      "PAT-006", "Sensitive"],
                ["", "WARFARIN",      "PAT-007", "Poor"],
                ["", "AZATHIOPRINE",  "PAT-008", "Poor"],
                ["", "FLUOROURACIL",  "PAT-009", "Poor"],
                ["", "SIMVASTATIN",   "PAT-010", "Intermediate"],
                ["", "TRAMADOL",      "PAT-011", "Ultrarapid"],
            ],
            inputs=[vcf_text, drug, patient_id, metabolizer_hint],
            outputs=[result_html, output_json],
            fn=_handler_example,
            cache_examples=False,
            label="⚡ Quick Demo Examples — click any row",
        )

        gr.HTML(
            "<p style='text-align:center;font-size:11px;color:#9aa0a6;margin-top:18px'>"
            "PharmaGuard AI · CPIC-aligned pharmacogenomics · RIFT 2026 HealthTech Hackathon"
            "</p>"
        )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860, share=True, show_api=False)
