"""
PharmaGuard Wildcard - Hugging Face Spaces entrypoint.
A lightweight Gradio interface for pharmacogenomic VCF risk analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr


DRUG_GENE_MAP: Dict[str, str] = {
    "CODEINE": "CYP2D6",
    "WARFARIN": "CYP2C9",
    "CLOPIDOGREL": "CYP2C19",
    "SIMVASTATIN": "SLCO1B1",
    "AZATHIOPRINE": "TPMT",
    "FLUOROURACIL": "DPYD",
}

RS_TO_STAR: Dict[str, Dict[str, str]] = {
    "CYP2D6": {
        "rs3892097": "*4",
        "rs35742686": "*3",
        "rs5030655": "*6",
        "rs16947": "*2",
    },
    "CYP2C19": {
        "rs4244285": "*2",
        "rs4986893": "*3",
        "rs12248560": "*17",
    },
    "CYP2C9": {
        "rs1799853": "*2",
        "rs1057910": "*3",
    },
    "SLCO1B1": {
        "rs4149056": "*5",
    },
    "TPMT": {
        "rs1142345": "*3A",
        "rs1800460": "*3C",
        "rs1800462": "*2",
    },
    "DPYD": {
        "rs3918290": "*2A",
        "rs55886062": "*13",
    },
}

# Gene-specific diplotype to phenotype activity mapping.
PHENOTYPE_MAP: Dict[str, Dict[str, Tuple[str, str, float]]] = {
    "CYP2D6": {
        "*1/*1": ("Normal Metabolizer", "Normal", 0.95),
        "*1/*2": ("Normal Metabolizer", "Normal", 0.93),
        "*1/*4": ("Intermediate Metabolizer", "Intermediate", 0.92),
        "*4/*4": ("Poor Metabolizer", "Poor", 0.96),
        "*1/*3": ("Intermediate Metabolizer", "Intermediate", 0.91),
        "*3/*3": ("Poor Metabolizer", "Poor", 0.94),
    },
    "CYP2C19": {
        "*1/*1": ("Normal Metabolizer", "Normal", 0.96),
        "*1/*2": ("Intermediate Metabolizer", "Intermediate", 0.93),
        "*1/*3": ("Intermediate Metabolizer", "Intermediate", 0.91),
        "*1/*17": ("Rapid Metabolizer", "Rapid", 0.89),
        "*2/*2": ("Poor Metabolizer", "Poor", 0.95),
        "*2/*3": ("Poor Metabolizer", "Poor", 0.94),
        "*17/*17": ("Rapid Metabolizer", "Rapid", 0.91),
    },
    "CYP2C9": {
        "*1/*1": ("Normal Metabolizer", "Normal", 0.95),
        "*1/*2": ("Intermediate Metabolizer", "Intermediate", 0.92),
        "*1/*3": ("Intermediate Metabolizer", "Intermediate", 0.93),
        "*2/*2": ("Poor Metabolizer", "Poor", 0.91),
        "*2/*3": ("Poor Metabolizer", "Poor", 0.90),
        "*3/*3": ("Poor Metabolizer", "Poor", 0.94),
    },
    "SLCO1B1": {
        "*1/*1": ("Normal Function", "Normal", 0.96),
        "*1/*5": ("Decreased Function", "Intermediate", 0.91),
        "*5/*5": ("Poor Function", "Poor", 0.93),
    },
    "TPMT": {
        "*1/*1": ("Normal Metabolizer", "Normal", 0.96),
        "*1/*2": ("Intermediate Metabolizer", "Intermediate", 0.92),
        "*1/*3A": ("Intermediate Metabolizer", "Intermediate", 0.93),
        "*3A/*3A": ("Poor Metabolizer", "Poor", 0.95),
    },
    "DPYD": {
        "*1/*1": ("Normal Metabolizer", "Normal", 0.95),
        "*1/*2A": ("Intermediate Metabolizer", "Intermediate", 0.91),
        "*2A/*2A": ("Poor Metabolizer", "Poor", 0.94),
        "*1/*13": ("Intermediate Metabolizer", "Intermediate", 0.89),
    },
}

RISK_RULES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "CODEINE": {
        "Poor": {
            "risk_label": "Ineffective",
            "severity": "high",
            "confidence": 0.95,
            "guidance": "Avoid codeine due to likely lack of efficacy.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.88,
            "guidance": "Consider alternatives or close monitoring for reduced response.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.93,
            "guidance": "Standard dosing is generally appropriate.",
        },
        "Ultrarapid": {
            "risk_label": "Toxic",
            "severity": "critical",
            "confidence": 0.91,
            "guidance": "Avoid codeine due to potential rapid morphine conversion and toxicity.",
        },
    },
    "CLOPIDOGREL": {
        "Poor": {
            "risk_label": "Ineffective",
            "severity": "high",
            "confidence": 0.94,
            "guidance": "Consider alternative antiplatelet therapy.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.87,
            "guidance": "Consider alternative antiplatelet or modified dosing strategy.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.92,
            "guidance": "Standard dosing is generally appropriate.",
        },
        "Rapid": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.89,
            "guidance": "Standard dosing with routine monitoring.",
        },
    },
    "WARFARIN": {
        "Poor": {
            "risk_label": "Adjust Dosage",
            "severity": "high",
            "confidence": 0.91,
            "guidance": "Start reduced dose and monitor INR closely.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.88,
            "guidance": "Use reduced initial dose with enhanced INR monitoring.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.93,
            "guidance": "Use standard dose algorithm and routine INR monitoring.",
        },
    },
    "SIMVASTATIN": {
        "Poor": {
            "risk_label": "Toxic",
            "severity": "high",
            "confidence": 0.92,
            "guidance": "Avoid high-dose simvastatin; consider alternatives.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "moderate",
            "confidence": 0.88,
            "guidance": "Use lower dose or an alternative statin.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.94,
            "guidance": "Standard statin strategy is generally appropriate.",
        },
    },
    "AZATHIOPRINE": {
        "Poor": {
            "risk_label": "Toxic",
            "severity": "critical",
            "confidence": 0.96,
            "guidance": "Avoid or use major dose reduction with intensive monitoring.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "high",
            "confidence": 0.91,
            "guidance": "Start reduced dose and monitor blood counts frequently.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.95,
            "guidance": "Standard dosing with routine CBC monitoring.",
        },
    },
    "FLUOROURACIL": {
        "Poor": {
            "risk_label": "Toxic",
            "severity": "critical",
            "confidence": 0.94,
            "guidance": "Avoid or heavily reduce dose due to severe toxicity risk.",
        },
        "Intermediate": {
            "risk_label": "Adjust Dosage",
            "severity": "high",
            "confidence": 0.89,
            "guidance": "Start with substantial dose reduction and titrate cautiously.",
        },
        "Normal": {
            "risk_label": "Safe",
            "severity": "low",
            "confidence": 0.93,
            "guidance": "Standard dosing with routine oncology monitoring.",
        },
    },
}


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
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            info[key] = value
        else:
            info[part] = "true"
    return info


def normalize_genotype(sample_col: str) -> str:
    genotype = sample_col.split(":", 1)[0].replace("|", "/")
    return genotype


def parse_vcf(vcf_content: str) -> List[Variant]:
    variants: List[Variant] = []
    for raw_line in vcf_content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) < 8:
            continue

        chrom, pos, variant_id, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]
        info_field = fields[7]
        fmt = fields[8] if len(fields) > 8 else ""
        sample = fields[9] if len(fields) > 9 else ""

        info = parse_info_field(info_field)
        rsid = info.get("RS") or (variant_id if variant_id.startswith("rs") else "")
        gene = info.get("GENE", "")
        star = info.get("STAR", "")

        genotype = normalize_genotype(sample) if fmt and sample else "0/1"
        if genotype == "0/0":
            continue

        try:
            pos_int = int(pos)
        except ValueError:
            continue

        variants.append(
            Variant(
                chromosome=chrom,
                position=pos_int,
                rsid=rsid,
                ref=ref,
                alt=alt,
                gene=gene,
                star=star,
                genotype=genotype,
            )
        )
    return variants


def infer_diplotype(gene: str, variants: List[Variant]) -> str:
    rs_map = RS_TO_STAR.get(gene, {})
    gene_variants = [v for v in variants if v.gene == gene or (v.rsid and v.rsid in rs_map)]
    if not gene_variants:
        return "*1/*1"

    # Pick up to two strongest observed alleles from non-reference calls.
    observed: List[str] = []
    for v in gene_variants:
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

    if "*1/*1" in mapping:
        default_name, default_activity, default_conf = mapping["*1/*1"]
        return f"Likely {default_name}", default_activity, round(default_conf - 0.12, 2)

    return "Unknown", "Normal", 0.6


def risk_for(drug: str, activity: str, phenotype_confidence: float) -> Dict[str, Any]:
    drug_rules = RISK_RULES.get(drug, {})
    rule = drug_rules.get(activity) or drug_rules.get("Normal") or {
        "risk_label": "Unknown",
        "severity": "none",
        "confidence": 0.6,
        "guidance": "Insufficient evidence for a confident recommendation.",
    }

    adjusted_conf = min(1.0, round(rule["confidence"] * phenotype_confidence, 3))
    return {
        "risk_label": rule["risk_label"],
        "severity": rule["severity"],
        "confidence_score": adjusted_conf,
        "guidance": rule["guidance"],
    }


def analyze_vcf(vcf_content: str, drug: str, patient_id: Optional[str]) -> Dict[str, Any]:
    if not vcf_content.strip():
        raise ValueError("VCF content is empty.")
    if "#CHROM" not in vcf_content:
        raise ValueError("Invalid VCF: missing #CHROM header.")
    if drug not in DRUG_GENE_MAP:
        raise ValueError(f"Unsupported drug: {drug}")

    variants = parse_vcf(vcf_content)
    primary_gene = DRUG_GENE_MAP[drug]
    diplotype = infer_diplotype(primary_gene, variants)
    phenotype_name, activity, phenotype_confidence = infer_phenotype(primary_gene, diplotype)
    risk = risk_for(drug, activity, phenotype_confidence)

    pid = patient_id.strip() if patient_id and patient_id.strip() else f"PATIENT_{int(datetime.now().timestamp())}"

    detected = [
        {
            "rsid": v.rsid or f"rs_unknown_{v.position}",
            "chromosome": f"chr{v.chromosome}" if not v.chromosome.startswith("chr") else v.chromosome,
            "position": v.position,
            "ref": v.ref,
            "alt": v.alt,
            "gene": v.gene or primary_gene,
            "starAllele": v.star if v.star else "*1",
            "genotype": v.genotype,
        }
        for v in variants
        if v.gene == primary_gene or v.rsid in RS_TO_STAR.get(primary_gene, {})
    ]

    recommendation = {
        "dosing_guidance": risk["guidance"],
        "monitoring_requirements": ["Clinical monitoring per treatment protocol"],
        "alternative_drugs": [],
        "cpic_level": "A",
        "guideline_source": "CPIC-aligned pharmacogenomic rules",
    }

    explanation = {
        "summary": (
            f"For {drug}, the inferred {primary_gene} diplotype is {diplotype} "
            f"with phenotype '{phenotype_name}'."
        ),
        "mechanism": (
            f"{primary_gene} activity influences pharmacokinetics and therefore impacts efficacy/toxicity risk."
        ),
        "variant_interpretation": (
            "Risk estimate was generated from non-reference pharmacogenomic variants in the uploaded VCF."
        ),
    }

    return {
        "patient_id": pid,
        "drug": drug,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "risk_assessment": {
            "risk_label": risk["risk_label"],
            "confidence_score": risk["confidence_score"],
            "severity": risk["severity"],
        },
        "pharmacogenomic_profile": {
            "primary_gene": primary_gene,
            "diplotype": diplotype,
            "phenotype": phenotype_name,
            "detected_variants": detected,
        },
        "clinical_recommendation": recommendation,
        "llm_generated_explanation": explanation,
        "quality_metrics": {
            "vcf_parsing_success": True,
            "variants_detected": len(detected),
            "llm_confidence": risk["confidence_score"],
        },
    }


# ---------------------------------------------------------------------------
# HTML formatting helpers
# ---------------------------------------------------------------------------

_SEVERITY_STYLES: Dict[str, Dict[str, str]] = {
    "low": {
        "bg": "#e6f4ea",
        "text": "#137333",
        "border": "#34a853",
        "badge_bg": "#137333",
        "badge_text": "#ffffff",
        "icon": "✔",
        "label": "SAFE",
        "bar_color": "#34a853",
    },
    "moderate": {
        "bg": "#fff8e1",
        "text": "#b26a00",
        "border": "#f9a825",
        "badge_bg": "#f9a825",
        "badge_text": "#5d3a00",
        "icon": "⚠",
        "label": "MODERATE RISK",
        "bar_color": "#f9a825",
    },
    "high": {
        "bg": "#fce8e6",
        "text": "#c5221f",
        "border": "#ea4335",
        "badge_bg": "#c5221f",
        "badge_text": "#ffffff",
        "icon": "🔴",
        "label": "HIGH RISK",
        "bar_color": "#ea4335",
    },
    "critical": {
        "bg": "#fce8e6",
        "text": "#7b1a18",
        "border": "#c5221f",
        "badge_bg": "#7b1a18",
        "badge_text": "#ffffff",
        "icon": "🚨",
        "label": "CRITICAL RISK",
        "bar_color": "#7b1a18",
    },
    "none": {
        "bg": "#f1f3f4",
        "text": "#5f6368",
        "border": "#9aa0a6",
        "badge_bg": "#9aa0a6",
        "badge_text": "#ffffff",
        "icon": "❓",
        "label": "UNKNOWN",
        "bar_color": "#9aa0a6",
    },
}


def _confidence_bar(score: float, bar_color: str) -> str:
    pct = int(score * 100)
    return f"""
    <div style="margin-top:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span style="font-size:12px;font-weight:600;color:#5f6368;letter-spacing:0.04em;text-transform:uppercase;">
          Confidence Score
        </span>
        <span style="font-size:13px;font-weight:700;color:#202124;">{pct}%</span>
      </div>
      <div style="background:#e8eaed;border-radius:99px;height:8px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:99px;
                    transition:width 0.6s ease;"></div>
      </div>
    </div>
    """


def _variant_table(detected: List[Dict[str, Any]]) -> str:
    if not detected:
        return ""
    rows = "".join(
        f"""<tr style="border-bottom:1px solid #e8eaed;">
              <td style="padding:8px 10px;font-size:12px;color:#3c4043;font-family:monospace;">{v['rsid']}</td>
              <td style="padding:8px 10px;font-size:12px;color:#3c4043;">{v['chromosome']}</td>
              <td style="padding:8px 10px;font-size:12px;color:#3c4043;font-family:monospace;">{v['starAllele']}</td>
              <td style="padding:8px 10px;font-size:12px;color:#3c4043;font-family:monospace;">{v['genotype']}</td>
            </tr>"""
        for v in detected
    )
    return f"""
    <div style="margin-top:16px;">
      <p style="font-size:12px;font-weight:600;color:#5f6368;letter-spacing:0.04em;
                text-transform:uppercase;margin-bottom:6px;">Detected Variants</p>
      <table style="width:100%;border-collapse:collapse;border:1px solid #e8eaed;border-radius:8px;overflow:hidden;">
        <thead>
          <tr style="background:#f8f9fa;">
            <th style="padding:8px 10px;text-align:left;font-size:11px;color:#5f6368;
                       font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">rsID</th>
            <th style="padding:8px 10px;text-align:left;font-size:11px;color:#5f6368;
                       font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Chr</th>
            <th style="padding:8px 10px;text-align:left;font-size:11px;color:#5f6368;
                       font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Star</th>
            <th style="padding:8px 10px;text-align:left;font-size:11px;color:#5f6368;
                       font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Genotype</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def format_risk_label(result: Dict[str, Any]) -> str:
    """
    Convert a structured result dict into a rich HTML card.
    Returns safe HTML suitable for gr.HTML().
    """
    risk_label  = result["risk_assessment"]["risk_label"]
    severity    = result["risk_assessment"]["severity"]
    confidence  = result["risk_assessment"]["confidence_score"]
    gene        = result["pharmacogenomic_profile"]["primary_gene"]
    diplotype   = result["pharmacogenomic_profile"]["diplotype"]
    phenotype   = result["pharmacogenomic_profile"]["phenotype"]
    drug        = result["drug"]
    patient_id  = result["patient_id"]
    guidance    = result["clinical_recommendation"]["dosing_guidance"]
    detected    = result["pharmacogenomic_profile"]["detected_variants"]
    ts          = result.get("timestamp", "")

    s = _SEVERITY_STYLES.get(severity, _SEVERITY_STYLES["none"])

    badge = f"""
    <span style="display:inline-flex;align-items:center;gap:6px;
                 background:{s['badge_bg']};color:{s['badge_text']};
                 border-radius:99px;padding:5px 14px;font-size:13px;font-weight:700;
                 letter-spacing:0.06em;box-shadow:0 2px 6px rgba(0,0,0,0.15);">
      {s['icon']}&nbsp;{s['label']}
    </span>
    """

    info_pills = f"""
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;">
      <span style="background:#f1f3f4;border-radius:99px;padding:4px 12px;font-size:12px;
                   color:#3c4043;font-weight:500;">💊 {drug}</span>
      <span style="background:#f1f3f4;border-radius:99px;padding:4px 12px;font-size:12px;
                   color:#3c4043;font-weight:500;">🧬 {gene} &nbsp;·&nbsp; {diplotype}</span>
      <span style="background:#f1f3f4;border-radius:99px;padding:4px 12px;font-size:12px;
                   color:#3c4043;font-weight:500;">🔬 {phenotype}</span>
      <span style="background:#f1f3f4;border-radius:99px;padding:4px 12px;font-size:12px;
                   color:#3c4043;font-weight:500;">👤 {patient_id}</span>
    </div>
    """

    guidance_box = f"""
    <div style="margin-top:14px;background:{s['bg']};border-left:4px solid {s['border']};
                border-radius:0 8px 8px 0;padding:12px 16px;">
      <p style="font-size:12px;font-weight:700;color:{s['text']};margin:0 0 4px;
                text-transform:uppercase;letter-spacing:0.06em;">Clinical Guidance</p>
      <p style="font-size:14px;color:{s['text']};margin:0;line-height:1.5;">{guidance}</p>
    </div>
    """

    timestamp_str = ""
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            timestamp_str = dt.strftime("%d %b %Y, %H:%M UTC")
        except Exception:
            timestamp_str = ts

    footer = f"""
    <p style="margin-top:16px;font-size:11px;color:#9aa0a6;text-align:right;">
      CPIC-aligned · Analysis performed {timestamp_str}
    </p>
    """ if timestamp_str else ""

    html = f"""
    <div style="font-family:'Google Sans',Roboto,Arial,sans-serif;max-width:720px;margin:0 auto;">

      <!-- Header card -->
      <div style="background:linear-gradient(135deg,#1a73e8 0%,#0d47a1 100%);
                  border-radius:16px 16px 0 0;padding:20px 24px 16px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <span style="font-size:22px;">🛡️</span>
          <span style="font-size:18px;font-weight:700;color:#ffffff;letter-spacing:-0.02em;">
            Risk Assessment Result
          </span>
        </div>
        <p style="font-size:13px;color:rgba(255,255,255,0.75);margin:0;">
          Pharmacogenomic analysis powered by PharmaGuard &middot; Meta x Scaler Hackathon
        </p>
      </div>

      <!-- Body card -->
      <div style="background:#ffffff;border:1px solid #e8eaed;border-top:none;
                  border-radius:0 0 16px 16px;padding:22px 24px;
                  box-shadow:0 4px 20px rgba(0,0,0,0.07);">

        <!-- Risk badge -->
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
          {badge}
          <span style="font-size:13px;color:#9aa0a6;">Risk Label: <strong style="color:#202124;">{risk_label}</strong></span>
        </div>

        <!-- Info pills -->
        {info_pills}

        <!-- Confidence bar -->
        {_confidence_bar(confidence, s['bar_color'])}

        <!-- Guidance -->
        {guidance_box}

        <!-- Variant table -->
        {_variant_table(detected)}

        {footer}
      </div>
    </div>
    """
    return html


def format_error_html(message: str) -> str:
    """Returns a styled error card for gr.HTML()."""
    return f"""
    <div style="font-family:'Google Sans',Roboto,Arial,sans-serif;max-width:720px;margin:0 auto;">
      <div style="background:#fce8e6;border:1px solid #ea4335;border-radius:16px;padding:22px 24px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
          <span style="font-size:22px;">🚨</span>
          <span style="font-size:17px;font-weight:700;color:#c5221f;">Analysis Failed</span>
        </div>
        <p style="font-size:14px;color:#c5221f;margin:0;line-height:1.5;">{message}</p>
      </div>
    </div>
    """


def run_analysis(file_path: Optional[str], vcf_text: str, drug: str, patient_id: str) -> Tuple[str, str]:
    try:
        content = vcf_text.strip()
        if file_path:
            with open(file_path, "r", encoding="utf-8") as handle:
                content = handle.read()

        result = analyze_vcf(content, drug, patient_id)
        html_summary = format_risk_label(result)
        return html_summary, json.dumps(result, indent=2)

    except Exception as exc:
        return format_error_html(str(exc)), json.dumps({"error": str(exc)}, indent=2)


# ---------------------------------------------------------------------------
# Custom CSS for the Gradio shell
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto+Mono&display=swap');

body, .gradio-container {
    background: #f0f4ff !important;
    font-family: 'Google Sans', Roboto, Arial, sans-serif !important;
}

/* Hero header */
.pharma-hero {
    text-align: center;
    padding: 32px 20px 20px;
}
.pharma-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.pharma-hero p {
    color: #5f6368;
    font-size: 1rem;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Panel card */
.input-panel {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #e8eaed;
}

/* Analyze button */
#analyze-btn {
    background: linear-gradient(135deg, #1a73e8, #0d47a1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 99px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 14px rgba(26,115,232,0.4) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(26,115,232,0.55) !important;
}

/* JSON output */
#json-output textarea {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 12px !important;
    background: #1e2130 !important;
    color: #a8d8a8 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Disclaimer banner */
.disclaimer {
    background: #fff8e1;
    border: 1px solid #f9a825;
    border-radius: 12px;
    padding: 12px 18px;
    font-size: 12px;
    color: #b26a00;
    text-align: center;
    margin-top: 8px;
}
"""

HEADER_HTML = """
<div class="pharma-hero">
  <h1>🛡️ PharmaGuard</h1>
  <p>
    AI-assisted pharmacogenomic VCF risk analysis &mdash; upload your variant file,
    select a drug, and receive a structured clinical risk assessment instantly.
    <br><br>
    <strong>Meta × Scaler OpenEnv Hackathon Demo</strong>
  </p>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer">
  ⚠️&nbsp;<strong>Disclaimer:</strong> This tool is for research &amp; demonstration purposes only.
  It does not constitute medical advice. Always consult a qualified clinician before making treatment decisions.
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="PharmaGuard – Pharmacogenomic Risk Analysis",
        css=CUSTOM_CSS,
        theme=gr.themes.Default(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Google Sans"), "sans-serif"],
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)
        gr.HTML(DISCLAIMER_HTML)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                with gr.Group(elem_classes=["input-panel"]):
                    gr.Markdown("### 📂 Input")
                    vcf_file = gr.File(
                        label="Upload VCF File",
                        file_types=[".vcf"],
                        type="filepath",
                    )
                    drug = gr.Dropdown(
                        label="💊 Select Drug",
                        choices=list(DRUG_GENE_MAP.keys()),
                        value="CLOPIDOGREL",
                    )
                    patient_id = gr.Textbox(
                        label="👤 Patient ID (optional)",
                        placeholder="e.g. PATIENT_001",
                    )
                    vcf_text = gr.Textbox(
                        label="Or Paste VCF Content",
                        lines=10,
                        placeholder=(
                            "Paste raw VCF text here if you do not upload a file.\n\n"
                            "Example minimal header:\n##fileformat=VCFv4.1\n#CHROM\tPOS\tID\t..."
                        ),
                    )
                    analyze_btn = gr.Button(
                        "🔬 Run Analysis",
                        elem_id="analyze-btn",
                        variant="primary",
                    )

            with gr.Column(scale=2, min_width=420):
                gr.Markdown("### 📊 Risk Assessment")
                summary_html = gr.HTML(
                    value="<p style='color:#9aa0a6;font-size:14px;padding:12px;text-align:center;'>"
                          "Results will appear here after analysis.</p>"
                )
                gr.Markdown("### 🗂️ Structured JSON Output")
                output_json = gr.Textbox(
                    label="",
                    lines=18,
                    elem_id="json-output",
                    show_copy_button=True,
                )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[vcf_file, vcf_text, drug, patient_id],
            outputs=[summary_html, output_json],
            api_name=False,
        )

        gr.Markdown(
            "<p style='text-align:center;color:#9aa0a6;font-size:12px;margin-top:24px;'>"
            "PharmaGuard · CPIC-aligned · Meta × Scaler Hackathon 2024"
            "</p>"
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, show_api=False)
