"""
Generates a one-page PDF report summarising the agent's run:
problem type, best model, metrics, top features (with chart), and the
plain-language explanation from the LLM.
"""
import os
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_report(
    output_path: str,
    job_id: str,
    plan: Dict[str, Any],
    best_model_name: str,
    metrics: Dict[str, float],
    top_features: List[Dict[str, Any]],
    explanation_text: str,
    shap_chart_path: str,
) -> str:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    story = []

    story.append(Paragraph("AutoDS Agent -- Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Run ID:</b> {job_id}", styles["Normal"]))
    story.append(Paragraph(f"<b>Problem type:</b> {plan['problem_type']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Target column:</b> {plan['target_column']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Best model:</b> {best_model_name}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Performance metrics", styles["Heading2"]))
    metric_rows = [["Metric", "Value"]] + [[k, f"{v:.4f}"] for k, v in metrics.items()]
    table = Table(metric_rows, colWidths=[6 * cm, 6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("What drives the model's predictions", styles["Heading2"]))
    story.append(Paragraph(explanation_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    if os.path.exists(shap_chart_path):
        story.append(Image(shap_chart_path, width=14 * cm, height=8 * cm))

    doc.build(story)
    return output_path
