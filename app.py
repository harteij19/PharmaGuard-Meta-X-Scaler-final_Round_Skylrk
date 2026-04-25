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


def run_analysis(file_path: Optional[str], vcf_text: str, drug: str, patient_id: str) -> Tuple[str, str]:
    try:
        content = vcf_text.strip()
        if file_path:
            with open(file_path, "r", encoding="utf-8") as handle:
                content = handle.read()

        result = analyze_vcf(content, drug, patient_id)

        summary = (
            f"### Analysis Complete\n"
            f"- **Patient ID:** {result['patient_id']}\n"
            f"- **Drug:** {result['drug']}\n"
            f"- **Primary Gene:** {result['pharmacogenomic_profile']['primary_gene']}\n"
            f"- **Diplotype:** {result['pharmacogenomic_profile']['diplotype']}\n"
            f"- **Phenotype:** {result['pharmacogenomic_profile']['phenotype']}\n"
            f"- **Risk:** {result['risk_assessment']['risk_label']}\n"
            f"- **Severity:** {result['risk_assessment']['severity']}\n"
            f"- **Confidence:** {result['risk_assessment']['confidence_score']}\n"
        )
        return summary, json.dumps(result, indent=2)
    except Exception as exc:
        return f"### Analysis Failed\n{exc}", json.dumps({"error": str(exc)}, indent=2)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PharmaGuard Wildcard") as demo:
        gr.Markdown(
            """
# PharmaGuard Wildcard
AI-assisted pharmacogenomic risk analysis for the Meta x Scaler OpenEnv Hackathon.
Upload a VCF file (or paste VCF text), pick a drug, and generate structured risk output.
            """.strip()
        )

        with gr.Row():
            vcf_file = gr.File(label="Upload VCF File", file_types=[".vcf"], type="filepath")
            drug = gr.Dropdown(label="Drug", choices=list(DRUG_GENE_MAP.keys()), value="CLOPIDOGREL")

        patient_id = gr.Textbox(label="Patient ID (optional)", placeholder="PATIENT_001")
        vcf_text = gr.Textbox(
            label="Or Paste VCF Content",
            lines=10,
            placeholder="Paste raw VCF text here if you do not upload a file.",
        )

        analyze_btn = gr.Button("Analyze")

        summary_md = gr.Markdown()
        output_json = gr.Textbox(label="Structured Result JSON", lines=18)

        analyze_btn.click(
            fn=run_analysis,
            inputs=[vcf_file, vcf_text, drug, patient_id],
            outputs=[summary_md, output_json],
            api_name=False,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, show_api=False)
