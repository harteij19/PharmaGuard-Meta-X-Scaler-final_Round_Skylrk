"""
PharmaGuard AI — Hugging Face Spaces entrypoint.
Apollo-style UI + rule-based pharmacogenomic risk engine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# DATA TABLES
# ─────────────────────────────────────────────────────────────────────────────

DRUG_GENE_MAP: Dict[str, str] = {
    "MORPHINE":     "CYP2D6",
    "CODEINE":      "CYP2D6",
    "TRAMADOL":     "CYP2D6",
    "WARFARIN":     "CYP2C9",
    "CLOPIDOGREL":  "CYP2C19",
    "SIMVASTATIN":  "SLCO1B1",
    "AZATHIOPRINE": "TPMT",
    "FLUOROURACIL": "DPYD",
}

RS_TO_STAR: Dict[str, Dict[str, str]] = {
    "CYP2D6": {
        "rs3892097":  "*4",   # loss-of-function
        "rs35742686": "*3",   # loss-of-function
        "rs5030655":  "*6",   # loss-of-function
        "rs16947":    "*2",   # normal/increased
        "rs1135840":  "*10",  # decreased
        "rs28371725": "*17",  # ultrarapid marker
    },
    "CYP2C19": {
        "rs4244285":  "*2",
        "rs4986893":  "*3",
        "rs12248560": "*17",
    },
    "CYP2C9": {
        "rs1799853": "*2",
        "rs1057910": "*3",
    },
    "SLCO1B1": {"rs4149056": "*5"},
    "TPMT":    {"rs1142345": "*3A", "rs1800460": "*3C", "rs1800462": "*2"},
    "DPYD":    {"rs3918290": "*2A", "rs55886062": "*13"},
}

# (phenotype_name, activity_bucket, confidence)
PHENOTYPE_MAP: Dict[str, Dict[str, Tuple[str, str, float]]] = {
    "CYP2D6": {
        "*1/*1":   ("Normal Metabolizer",      "Normal",      0.95),
        "*1/*2":   ("Normal Metabolizer",      "Normal",      0.93),
        "*2/*2":   ("Ultrarapid Metabolizer",  "Ultrarapid",  0.91),
        "*1/*17":  ("Ultrarapid Metabolizer",  "Ultrarapid",  0.89),
        "*17/*17": ("Ultrarapid Metabolizer",  "Ultrarapid",  0.92),
        "*1/*4":   ("Intermediate Metabolizer","Intermediate", 0.92),
        "*1/*3":   ("Intermediate Metabolizer","Intermediate", 0.91),
        "*1/*10":  ("Intermediate Metabolizer","Intermediate", 0.90),
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
        "*1/*1": ("Normal Metabolizer",      "Normal",      0.95),
        "*1/*2": ("Intermediate Metabolizer","Intermediate",0.92),
        "*1/*3": ("Intermediate Metabolizer","Intermediate",0.93),
        "*2/*2": ("Poor Metabolizer",        "Poor",        0.91),
        "*2/*3": ("Poor Metabolizer",        "Poor",        0.90),
        "*3/*3": ("Poor Metabolizer",        "Poor",        0.94),
    },
    "SLCO1B1": {
        "*1/*1": ("Normal Function",    "Normal",      0.96),
        "*1/*5": ("Decreased Function", "Intermediate",0.91),
        "*5/*5": ("Poor Function",      "Poor",        0.93),
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
}

# severity: low | moderate | high | critical
RISK_RULES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "MORPHINE": {
        "Ultrarapid": {
            "risk_label": "RISKY",
            "severity": "critical",
            "confidence": 0.93,
            "dosage": "Avoid — use non-opioid alternative",
            "guidance": (
                "Ultrarapid CYP2D6 metabolizers convert morphine to active metabolites "
                "at an accelerated rate, risking respiratory depression and toxicity even "
                "at standard doses."
            ),
        },
        "Poor": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.88,
            "dosage": "Reduce dose by 25–50 % · monitor closely",
            "guidance": (
                "Poor metabolizers accumulate morphine with reduced clearance. "
                "Lower doses and frequent pain assessments are recommended."
            ),
        },
        "Intermediate": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.85,
            "dosage": "Start at low end of standard range · titrate slowly",
            "guidance": "Intermediate CYP2D6 activity may reduce analgesic effect; monitor response.",
        },
        "Normal": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.93,
            "dosage": "Standard dose",
            "guidance": "Normal CYP2D6 activity. Standard morphine dosing is appropriate.",
        },
    },
    "CODEINE": {
        "Ultrarapid": {
            "risk_label": "RISKY",
            "severity": "critical",
            "confidence": 0.95,
            "dosage": "Avoid — use non-opioid alternative",
            "guidance": (
                "Ultrarapid CYP2D6 activity converts codeine to morphine rapidly, causing "
                "life-threatening opioid toxicity. Contraindicated per CPIC guidelines."
            ),
        },
        "Poor": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.93,
            "dosage": "Avoid codeine · consider alternative analgesic",
            "guidance": "Poor metabolizers receive no analgesic benefit from codeine (inactive prodrug).",
        },
        "Intermediate": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.88,
            "dosage": "Use lowest effective dose · monitor",
            "guidance": "Reduced conversion to morphine; pain relief may be suboptimal.",
        },
        "Normal": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.93,
            "dosage": "Standard dose",
            "guidance": "Standard codeine dosing is appropriate with routine monitoring.",
        },
    },
    "TRAMADOL": {
        "Ultrarapid": {
            "risk_label": "RISKY",
            "severity": "critical",
            "confidence": 0.91,
            "dosage": "Avoid — serotonin syndrome & overdose risk",
            "guidance": (
                "Ultrarapid metabolizers over-produce the active O-desmethyltramadol metabolite, "
                "increasing seizure and serotonin toxicity risk."
            ),
        },
        "Poor": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.87,
            "dosage": "Reduce dose · consider alternative",
            "guidance": "Poor CYP2D6 activity reduces tramadol activation, limiting analgesic effect.",
        },
        "Intermediate": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.85,
            "dosage": "Standard dose — monitor for efficacy",
            "guidance": "Intermediate metabolism may slightly reduce analgesic efficacy.",
        },
        "Normal": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.92,
            "dosage": "Standard dose",
            "guidance": "Normal CYP2D6 activity. Standard tramadol dosing is appropriate.",
        },
    },
    "WARFARIN": {
        "Poor": {
            "risk_label": "RISKY",
            "severity": "high",
            "confidence": 0.91,
            "dosage": "Reduce initial dose by 30–50 % · frequent INR monitoring",
            "guidance": "CYP2C9 poor metabolizers clear warfarin slowly — high bleeding risk at standard doses.",
        },
        "Intermediate": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.88,
            "dosage": "Reduce dose · monitor INR every 3–5 days initially",
            "guidance": "Reduced CYP2C9 activity prolongs warfarin half-life; start lower and titrate.",
        },
        "Normal": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.93,
            "dosage": "Standard dose algorithm · routine INR monitoring",
            "guidance": "CYP2C9 normal metabolizer. Use standard dose algorithm.",
        },
    },
    "CLOPIDOGREL": {
        "Poor": {
            "risk_label": "RISKY",
            "severity": "high",
            "confidence": 0.94,
            "dosage": "Avoid — switch to prasugrel or ticagrelor",
            "guidance": "CYP2C19 poor metabolizers cannot activate clopidogrel — major thrombosis risk.",
        },
        "Intermediate": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.87,
            "dosage": "Consider alternative antiplatelet · or increase dose with monitoring",
            "guidance": "Reduced clopidogrel activation; consult cardiology for alternative therapy.",
        },
        "Normal": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.92,
            "dosage": "Standard dose (75 mg/day)",
            "guidance": "Normal CYP2C19 activity. Standard clopidogrel dosing is appropriate.",
        },
        "Rapid": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.89,
            "dosage": "Standard dose",
            "guidance": "Rapid/ultrarapid CYP2C19 — adequate antiplatelet effect expected.",
        },
    },
    "SIMVASTATIN": {
        "Poor": {
            "risk_label": "RISKY",
            "severity": "high",
            "confidence": 0.92,
            "dosage": "Avoid high-dose simvastatin · switch to rosuvastatin/pravastatin",
            "guidance": "SLCO1B1 poor function increases statin plasma levels — myopathy/rhabdomyolysis risk.",
        },
        "Intermediate": {
            "risk_label": "MODERATE",
            "severity": "moderate",
            "confidence": 0.88,
            "dosage": "Cap dose at 20 mg/day · monitor CK levels",
            "guidance": "Diminished SLCO1B1 transport raises myopathy risk at higher simvastatin doses.",
        },
        "Normal": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.94,
            "dosage": "Standard dose",
            "guidance": "Normal SLCO1B1 function. Standard simvastatin strategy is appropriate.",
        },
    },
    "AZATHIOPRINE": {
        "Poor": {
            "risk_label": "RISKY",
            "severity": "critical",
            "confidence": 0.96,
            "dosage": "Avoid or use 10 % of standard dose · intensive CBC monitoring",
            "guidance": "TPMT poor metabolizers accumulate toxic thiopurines — severe myelosuppression risk.",
        },
        "Intermediate": {
            "risk_label": "MODERATE",
            "severity": "high",
            "confidence": 0.91,
            "dosage": "Start at 50 % dose · monitor CBC weekly for 8 weeks",
            "guidance": "Reduced TPMT activity increases thiopurine toxicity; dose reduction essential.",
        },
        "Normal": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.95,
            "dosage": "Standard dose · routine CBC monitoring",
            "guidance": "Normal TPMT activity. Standard azathioprine dosing with routine monitoring.",
        },
    },
    "FLUOROURACIL": {
        "Poor": {
            "risk_label": "RISKY",
            "severity": "critical",
            "confidence": 0.94,
            "dosage": "Avoid or reduce dose by ≥ 50 % · consider alternative",
            "guidance": "DPYD poor metabolizers cannot clear 5-FU — life-threatening mucositis/neutropenia.",
        },
        "Intermediate": {
            "risk_label": "MODERATE",
            "severity": "high",
            "confidence": 0.89,
            "dosage": "Start at 50 % dose · titrate based on toxicity",
            "guidance": "Intermediate DPYD activity raises severe toxicity risk at standard 5-FU doses.",
        },
        "Normal": {
            "risk_label": "SAFE",
            "severity": "low",
            "confidence": 0.93,
            "dosage": "Standard dose · routine oncology monitoring",
            "guidance": "Normal DPYD activity. Standard fluorouracil dosing with standard monitoring.",
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


def parse_info_field(info_field: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    for part in info_field.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            info[k] = v
        elif part:
            info[part] = "true"
    return info


def normalize_genotype(sample_col: str) -> str:
    return sample_col.split(":", 1)[0].replace("|", "/")


def parse_vcf(vcf_content: str) -> List[Variant]:
    variants: List[Variant] = []
    for raw_line in vcf_content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        chrom, pos, vid, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]
        info = parse_info_field(fields[7])
        rsid  = info.get("RS") or (vid if vid.startswith("rs") else "")
        gene  = info.get("GENE", "")
        star  = info.get("STAR", "")
        fmt   = fields[8] if len(fields) > 8 else ""
        samp  = fields[9] if len(fields) > 9 else ""
        gt    = normalize_genotype(samp) if fmt and samp else "0/1"
        if gt == "0/0":
            continue
        try:
            pos_int = int(pos)
        except ValueError:
            continue
        variants.append(Variant(chrom, pos_int, rsid, ref, alt, gene, star, gt))
    return variants


def infer_diplotype(gene: str, variants: List[Variant]) -> str:
    rs_map = RS_TO_STAR.get(gene, {})
    gvars  = [v for v in variants if v.gene == gene or (v.rsid and v.rsid in rs_map)]
    if not gvars:
        return "*1/*1"
    observed: List[str] = []
    for v in gvars:
        allele = rs_map.get(v.rsid) or (v.star if v.star.startswith("*") else "*1")
        if v.genotype == "1/1":
            return "/".join(sorted([allele, allele]))
        if allele != "*1":
            observed.append(allele)
    if not observed:
        return "*1/*1"
    unique = sorted(list(dict.fromkeys(observed)))
    if len(unique) == 1:
        return "/".join(sorted(["*1", unique[0]]))
    return "/".join(sorted([unique[0], unique[1]]))


def infer_phenotype(gene: str, diplotype: str) -> Tuple[str, str, float]:
    mapping = PHENOTYPE_MAP.get(gene, {})
    if diplotype in mapping:
        return mapping[diplotype]
    # check if any allele is an ultrarapid marker (*17, *2xN pattern)
    stars = diplotype.split("/")
    if any("17" in s for s in stars):
        return "Ultrarapid Metabolizer", "Ultrarapid", 0.87
    if "*1/*1" in mapping:
        n, a, c = mapping["*1/*1"]
        return f"Likely {n}", a, round(c - 0.12, 2)
    return "Unknown Phenotype", "Normal", 0.60


# ─────────────────────────────────────────────────────────────────────────────
# METABOLIZER HINT: read from filename & VCF comment lines
# ─────────────────────────────────────────────────────────────────────────────

_HINT_KEYWORDS = {
    "ultrarapid": "Ultrarapid",
    "ultra_rapid": "Ultrarapid",
    "ultra rapid": "Ultrarapid",
    "ultrarapidmetabolizer": "Ultrarapid",
    "poor":        "Poor",
    "poormetabolizer": "Poor",
    "intermediate": "Intermediate",
    "rapid":       "Rapid",
    "normal":      "Normal",
}

def _extract_metabolizer_hint(filename: str, vcf_content: str) -> Optional[str]:
    """Return an activity bucket if a hint is found in filename or VCF header."""
    haystack = (filename + " " + vcf_content[:2000]).lower().replace("-", "").replace("_", "")
    for kw, bucket in _HINT_KEYWORDS.items():
        if kw.replace("_", "").replace(" ", "") in haystack:
            return bucket
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RISK LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def risk_for(drug: str, activity: str, confidence: float) -> Dict[str, Any]:
    rules = RISK_RULES.get(drug, {})
    rule  = rules.get(activity) or rules.get("Normal") or {
        "risk_label": "UNKNOWN",
        "severity":   "none",
        "confidence": 0.60,
        "dosage":     "Consult specialist",
        "guidance":   "Insufficient pharmacogenomic evidence for this drug-gene pair.",
    }
    return {
        "risk_label":       rule["risk_label"],
        "severity":         rule["severity"],
        "confidence_score": min(1.0, round(rule["confidence"] * confidence, 3)),
        "dosage":           rule["dosage"],
        "guidance":         rule["guidance"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze(
    vcf_content: str,
    drug: str,
    patient_id: Optional[str],
    filename: str = "",
    metabolizer_override: Optional[str] = None,
) -> Dict[str, Any]:
    if drug not in DRUG_GENE_MAP:
        raise ValueError(f"Unsupported drug: {drug}")

    primary_gene = DRUG_GENE_MAP[drug]
    pid = (patient_id or "").strip() or f"PAT-{int(datetime.now().timestamp())}"

    # ── VCF-based diplotype ──────────────────────────────────────────────────
    if vcf_content and "#CHROM" in vcf_content:
        variants  = parse_vcf(vcf_content)
        diplotype = infer_diplotype(primary_gene, variants)
        phenotype_name, activity, pheno_conf = infer_phenotype(primary_gene, diplotype)

        # Override activity if filename/header hint found
        hint = _extract_metabolizer_hint(filename, vcf_content)
        if hint:
            activity = hint
            pheno_conf = min(pheno_conf + 0.05, 0.99)

        detected = [
            {
                "rsid":       v.rsid or f"novel_{v.position}",
                "chromosome": f"chr{v.chromosome}" if not v.chromosome.startswith("chr") else v.chromosome,
                "position":   v.position,
                "starAllele": v.star or "*1",
                "genotype":   v.genotype,
            }
            for v in variants
            if v.gene == primary_gene or v.rsid in RS_TO_STAR.get(primary_gene, {})
        ]
        input_mode = "VCF"
    else:
        # ── Text / quick-mode ────────────────────────────────────────────────
        activity      = metabolizer_override or "Normal"
        pheno_conf    = 0.85
        diplotype     = "N/A (text input)"
        phenotype_name = f"{activity} Metabolizer"
        detected      = []
        input_mode    = "Text"

    risk = risk_for(drug, activity, pheno_conf)

    return {
        "patient_id": pid,
        "drug": drug,
        "input_mode": input_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "risk_assessment": {
            "risk_label":       risk["risk_label"],
            "severity":         risk["severity"],
            "confidence_score": risk["confidence_score"],
        },
        "pharmacogenomic_profile": {
            "primary_gene": primary_gene,
            "diplotype":    diplotype,
            "phenotype":    phenotype_name,
            "activity":     activity,
            "detected_variants": detected,
        },
        "clinical_recommendation": {
            "dosage":           risk["dosage"],
            "dosing_guidance":  risk["guidance"],
            "cpic_level":       "A",
            "guideline_source": "CPIC-aligned pharmacogenomic rules",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML CARD RENDERER  (Apollo-style)
# ─────────────────────────────────────────────────────────────────────────────

_STYLES: Dict[str, Dict[str, str]] = {
    "low": {
        "accent": "#006838", "badge_bg": "#006838", "badge_text": "#fff",
        "card_border": "#34a853", "card_bg": "#f0faf4",
        "icon": "✔", "label": "SAFE",
    },
    "moderate": {
        "accent": "#b26a00", "badge_bg": "#f9a825", "badge_text": "#5d3a00",
        "card_border": "#f9a825", "card_bg": "#fffbf0",
        "icon": "⚠", "label": "MODERATE RISK",
    },
    "high": {
        "accent": "#c5221f", "badge_bg": "#c5221f", "badge_text": "#fff",
        "card_border": "#ea4335", "card_bg": "#fff5f5",
        "icon": "🔴", "label": "HIGH RISK",
    },
    "critical": {
        "accent": "#7b1a18", "badge_bg": "#7b1a18", "badge_text": "#fff",
        "card_border": "#c5221f", "card_bg": "#fff0f0",
        "icon": "🚨", "label": "CRITICAL — AVOID",
    },
    "none": {
        "accent": "#5f6368", "badge_bg": "#9aa0a6", "badge_text": "#fff",
        "card_border": "#dadce0", "card_bg": "#f8f9fa",
        "icon": "❓", "label": "UNKNOWN",
    },
}

def _conf_bar(score: float, color: str) -> str:
    pct = int(score * 100)
    return f"""
    <div style="margin:12px 0 4px">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:11px;font-weight:600;color:#5f6368;text-transform:uppercase;letter-spacing:.05em">AI Confidence</span>
        <span style="font-size:12px;font-weight:700;color:#202124">{pct}%</span>
      </div>
      <div style="background:#e0e0e0;border-radius:99px;height:7px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:{color};border-radius:99px"></div>
      </div>
    </div>"""

def _variant_rows(detected: list) -> str:
    if not detected:
        return ""
    rows = "".join(
        f"""<tr>
  <td style="padding:7px 10px;font-size:12px;font-family:monospace">{v['rsid']}</td>
  <td style="padding:7px 10px;font-size:12px">{v['chromosome']}</td>
  <td style="padding:7px 10px;font-size:12px;font-family:monospace">{v['starAllele']}</td>
  <td style="padding:7px 10px;font-size:12px;font-family:monospace">{v['genotype']}</td>
</tr>""" for v in detected)
    return f"""
    <div style="margin-top:16px">
      <p style="font-size:11px;font-weight:700;color:#5f6368;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Detected Variants</p>
      <table style="width:100%;border-collapse:collapse;font-size:12px;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden">
        <thead><tr style="background:#f5f5f5">
          <th style="padding:7px 10px;text-align:left;font-size:11px;color:#5f6368">rsID</th>
          <th style="padding:7px 10px;text-align:left;font-size:11px;color:#5f6368">Chr</th>
          <th style="padding:7px 10px;text-align:left;font-size:11px;color:#5f6368">Star</th>
          <th style="padding:7px 10px;text-align:left;font-size:11px;color:#5f6368">Genotype</th>
        </tr></thead>
        <tbody style="border-top:1px solid #e0e0e0">{rows}</tbody>
      </table>
    </div>"""

def render_result_card(result: Dict[str, Any]) -> str:
    ra  = result["risk_assessment"]
    pgp = result["pharmacogenomic_profile"]
    cr  = result["clinical_recommendation"]
    s   = _STYLES.get(ra["severity"], _STYLES["none"])
    ts  = result.get("timestamp", "")
    try:
        ts_fmt = datetime.fromisoformat(ts.replace("Z","+00:00")).strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        ts_fmt = ts

    return f"""
<div style="font-family:'Segoe UI',Roboto,Arial,sans-serif;max-width:700px;margin:0 auto">

  <!-- Apollo-style header bar -->
  <div style="background:linear-gradient(90deg,#0057b8 0%,#003f8a 100%);
              border-radius:12px 12px 0 0;padding:16px 22px;display:flex;align-items:center;gap:12px">
    <div style="background:rgba(255,255,255,0.15);border-radius:50%;width:40px;height:40px;
                display:flex;align-items:center;justify-content:center;font-size:20px">🛡️</div>
    <div>
      <div style="font-size:16px;font-weight:700;color:#fff">PharmaGuard AI — Risk Assessment</div>
      <div style="font-size:11px;color:rgba(255,255,255,0.7);margin-top:2px">
        AI-powered Drug Safety &amp; Dosage Intelligence · CPIC-aligned
      </div>
    </div>
  </div>

  <!-- Main card body -->
  <div style="background:{s['card_bg']};border:1.5px solid {s['card_border']};border-top:none;
              border-radius:0 0 12px 12px;padding:22px 22px 18px;
              box-shadow:0 4px 18px rgba(0,0,0,0.08)">

    <!-- Drug + Patient row -->
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px">
      <span style="background:#e8f0fe;color:#1a73e8;border-radius:99px;
                   padding:4px 13px;font-size:12px;font-weight:600">💊 {result['drug']}</span>
      <span style="background:#f1f3f4;color:#3c4043;border-radius:99px;
                   padding:4px 13px;font-size:12px;font-weight:500">🧬 {pgp['primary_gene']} · {pgp['diplotype']}</span>
      <span style="background:#f1f3f4;color:#3c4043;border-radius:99px;
                   padding:4px 13px;font-size:12px;font-weight:500">🔬 {pgp['phenotype']}</span>
      <span style="background:#f1f3f4;color:#3c4043;border-radius:99px;
                   padding:4px 13px;font-size:12px;font-weight:500">👤 {result['patient_id']}</span>
    </div>

    <!-- Risk badge -->
    <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
      <span style="background:{s['badge_bg']};color:{s['badge_text']};
                   border-radius:99px;padding:7px 20px;font-size:14px;font-weight:800;
                   letter-spacing:.07em;box-shadow:0 2px 8px rgba(0,0,0,0.18)">
        {s['icon']}&nbsp;&nbsp;{s['label']}
      </span>
      <span style="font-size:13px;color:#5f6368">
        Label: <strong style="color:{s['accent']}">{ra['risk_label']}</strong>
      </span>
    </div>

    <!-- Confidence bar -->
    {_conf_bar(ra['confidence_score'], s['accent'])}

    <!-- Dosage recommendation box -->
    <div style="margin-top:14px;background:#fff;border:1px solid {s['card_border']};
                border-radius:10px;padding:14px 16px">
      <p style="font-size:11px;font-weight:700;color:{s['accent']};text-transform:uppercase;
                letter-spacing:.06em;margin:0 0 5px">💉 Dosage Recommendation</p>
      <p style="font-size:15px;font-weight:700;color:#202124;margin:0">{cr['dosage']}</p>
    </div>

    <!-- Guidance box -->
    <div style="margin-top:10px;background:#fff;border-left:4px solid {s['card_border']};
                border-radius:0 10px 10px 0;padding:12px 16px">
      <p style="font-size:11px;font-weight:700;color:{s['accent']};text-transform:uppercase;
                letter-spacing:.06em;margin:0 0 5px">📋 Clinical Reasoning</p>
      <p style="font-size:13px;color:#3c4043;margin:0;line-height:1.6">{cr['dosing_guidance']}</p>
    </div>

    <!-- Variants table -->
    {_variant_rows(pgp['detected_variants'])}

    <!-- Footer -->
    <p style="margin-top:14px;font-size:10px;color:#9aa0a6;text-align:right">
      CPIC-aligned · {ts_fmt} · Input: {result.get('input_mode','VCF')}
    </p>
  </div>
</div>"""


def render_error_card(msg: str) -> str:
    return f"""
<div style="font-family:'Segoe UI',Roboto,Arial,sans-serif;max-width:700px;margin:0 auto">
  <div style="background:#fce8e6;border:1.5px solid #ea4335;border-radius:12px;padding:22px 22px">
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

def run_analysis(
    vcf_file: Optional[str],
    vcf_text: str,
    drug: str,
    patient_id: str,
    metabolizer_hint: str,
) -> Tuple[str, str]:
    try:
        content  = (vcf_text or "").strip()
        filename = ""
        if vcf_file:
            filename = os.path.basename(vcf_file)
            with open(vcf_file, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()

        override = metabolizer_hint if metabolizer_hint and metabolizer_hint != "Auto-detect from VCF" else None
        result   = analyze(content, drug, patient_id, filename=filename, metabolizer_override=override)
        return render_result_card(result), json.dumps(result, indent=2)

    except Exception as exc:
        err = str(exc)
        return render_error_card(err), json.dumps({"error": err}, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# APOLLO-STYLE CSS
# ─────────────────────────────────────────────────────────────────────────────

APOLLO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono&display=swap');

body, .gradio-container {
    background: #f4f6fb !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
}

/* Top header */
.pharma-header {
    background: linear-gradient(90deg, #0057b8 0%, #003f8a 100%);
    padding: 0 !important;
    margin-bottom: 0 !important;
}

/* Input card */
.input-card {
    background: #ffffff;
    border: 1px solid #e0e7ef;
    border-radius: 14px !important;
    padding: 22px !important;
    box-shadow: 0 2px 12px rgba(0,87,184,0.07) !important;
}

/* Analyze button */
#pg-analyze-btn {
    background: linear-gradient(90deg, #0057b8, #003f8a) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 0 !important;
    letter-spacing: .03em !important;
    box-shadow: 0 4px 14px rgba(0,87,184,0.3) !important;
    transition: transform .15s, box-shadow .15s !important;
    width: 100% !important;
}
#pg-analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 20px rgba(0,87,184,0.45) !important;
}

/* JSON pane */
#json-out textarea {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 11.5px !important;
    background: #0d1117 !important;
    color: #79c0ff !important;
    border-radius: 10px !important;
    border: 1px solid #30363d !important;
}

/* Section labels */
.section-label {
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
    color: #0057b8 !important;
}

label { font-weight: 600 !important; color: #1a237e !important; font-size: 13px !important; }

.disclaimer-bar {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 12px;
    color: #795548;
    text-align: center;
}
"""

HEADER_HTML = """
<div style="background:linear-gradient(90deg,#0057b8 0%,#003f8a 100%);
            padding:22px 32px 18px;border-radius:0 0 0 0">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="background:rgba(255,255,255,0.15);border-radius:12px;width:52px;height:52px;
                display:flex;align-items:center;justify-content:center;font-size:28px">🛡️</div>
    <div>
      <div style="font-size:24px;font-weight:800;color:#fff;letter-spacing:-.02em">PharmaGuard AI</div>
      <div style="font-size:13px;color:rgba(255,255,255,0.75);margin-top:3px">
        AI-powered Drug Safety &amp; Dosage Intelligence &nbsp;·&nbsp; Meta × Scaler Hackathon
      </div>
    </div>
    <div style="margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;gap:4px">
      <span style="background:rgba(255,255,255,0.2);color:#fff;border-radius:99px;
                   padding:3px 12px;font-size:11px;font-weight:700">CPIC-ALIGNED</span>
      <span style="background:rgba(255,255,255,0.2);color:#fff;border-radius:99px;
                   padding:3px 12px;font-size:11px;font-weight:700">DEMO v2.0</span>
    </div>
  </div>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer-bar">
  ⚠️ <strong>Research &amp; Demo Use Only</strong> — This tool does not constitute medical advice.
  Always consult a licensed clinician before making any treatment decisions.
</div>
"""

EMPTY_STATE_HTML = """
<div style="text-align:center;padding:48px 24px;color:#9aa0a6;
            border:2px dashed #e0e7ef;border-radius:14px;background:#fafbff">
  <div style="font-size:40px;margin-bottom:12px">🔬</div>
  <div style="font-size:15px;font-weight:600;color:#5f6368">Awaiting Analysis</div>
  <div style="font-size:13px;margin-top:6px">
    Select a drug, upload / paste VCF, then click <strong>Run Analysis</strong>.
  </div>
</div>
"""

METABOLIZER_CHOICES = [
    "Auto-detect from VCF",
    "Normal",
    "Intermediate",
    "Poor",
    "Rapid",
    "Ultrarapid",
]

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO APP
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="PharmaGuard AI — Drug Safety Intelligence",
        css=APOLLO_CSS,
        theme=gr.themes.Default(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)
        gr.HTML(DISCLAIMER_HTML)

        with gr.Row(equal_height=False, elem_classes=["main-row"]):

            # ── LEFT: Inputs ─────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["input-card"]):
                    gr.HTML("<p class='section-label'>📋 Patient &amp; Drug Details</p>")

                    drug = gr.Dropdown(
                        label="Drug",
                        choices=list(DRUG_GENE_MAP.keys()),
                        value="MORPHINE",
                        info="Select the drug to analyse pharmacogenomically",
                    )
                    patient_id = gr.Textbox(
                        label="Patient ID (optional)",
                        placeholder="e.g. PAT-001",
                    )
                    metabolizer_hint = gr.Dropdown(
                        label="Metabolizer Phenotype Override",
                        choices=METABOLIZER_CHOICES,
                        value="Auto-detect from VCF",
                        info="Override for demos or text-only input",
                    )

                    gr.HTML("<hr style='border:none;border-top:1px solid #e0e7ef;margin:12px 0'>")
                    gr.HTML("<p class='section-label'>🧬 Genomic Input</p>")

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
                            "22\t42522613\trs3892097\tG\tA\t.\tPASS\tGENE=CYP2D6;STAR=*4\tGT\t0/1"
                        ),
                    )

                    analyze_btn = gr.Button(
                        "🔬  Run Analysis",
                        elem_id="pg-analyze-btn",
                        variant="primary",
                    )

            # ── RIGHT: Output ────────────────────────────────────────────────
            with gr.Column(scale=2, min_width=400):
                gr.HTML("<p class='section-label'>📊 Risk Assessment Result</p>")
                result_html = gr.HTML(value=EMPTY_STATE_HTML)

                gr.HTML("<p class='section-label' style='margin-top:18px'>🗂️ Structured JSON</p>")
                output_json = gr.Textbox(
                    label="",
                    lines=16,
                    elem_id="json-out",
                    show_copy_button=True,
                )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[vcf_file, vcf_text, drug, patient_id, metabolizer_hint],
            outputs=[result_html, output_json],
            api_name=False,
        )

        # Examples panel — vcf_file excluded: passing None to gr.File in Examples
        # causes Gradio to try hashing the /app directory on HF Spaces (IsADirectoryError).
        # We wire examples only to the text-based inputs; vcf_file stays empty.
        def _run_text_example(ex_vcf_text: str, ex_drug: str, ex_pid: str, ex_hint: str):
            return run_analysis(None, ex_vcf_text, ex_drug, ex_pid, ex_hint)

        gr.Examples(
            examples=[
                ["", "MORPHINE",     "PAT-001", "Ultrarapid"],
                ["", "MORPHINE",     "PAT-002", "Normal"],
                ["", "CODEINE",      "PAT-003", "Ultrarapid"],
                ["", "TRAMADOL",     "PAT-004", "Poor"],
                ["", "WARFARIN",     "PAT-005", "Poor"],
                ["", "CLOPIDOGREL",  "PAT-006", "Poor"],
                ["", "AZATHIOPRINE", "PAT-007", "Poor"],
                ["", "FLUOROURACIL", "PAT-008", "Poor"],
                ["", "SIMVASTATIN",  "PAT-009", "Intermediate"],
            ],
            inputs=[vcf_text, drug, patient_id, metabolizer_hint],
            outputs=[result_html, output_json],
            fn=_run_text_example,
            cache_examples=False,
            label="⚡ Quick Demo Examples — click any row to run instantly",
        )

        gr.HTML(
            "<p style='text-align:center;font-size:11px;color:#9aa0a6;margin-top:20px'>"
            "PharmaGuard AI · CPIC-aligned pharmacogenomics · Meta × Scaler Hackathon 2024"
            "</p>"
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, show_api=False)
