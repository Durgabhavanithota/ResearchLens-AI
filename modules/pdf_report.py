
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

from modules.paper_analyzer import analyze_paper


def generate_research_report(
    papers,
    research_area,
    comparison_result=None,
    gap_results=None
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        fontSize=22,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#176B87"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = styles["BodyText"]

    story = []

    def add_paragraph(text, style=body_style):
        safe_text = escape(str(text))
        story.append(Paragraph(safe_text, style))
        story.append(Spacer(1, 7))

    # Report title
    story.append(
        Paragraph("ResearchLens AI", title_style)
    )

    story.append(
        Paragraph(
            "AI-Powered Research Paper Comparison "
            "and Research Gap Detection System Using NLP",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph("Research Analysis Report", heading_style)
    )

    add_paragraph(f"Research Area: {research_area}")
    add_paragraph(f"Total Papers Analyzed: {len(papers)}")

    # Paper details
    story.append(
        Paragraph("1. Uploaded Research Papers", heading_style)
    )

    for index, paper in enumerate(papers, start=1):

        add_paragraph(f"{index}. {paper['name']}")

        analysis = analyze_paper(paper["text"])

        for field in [
            "Title",
            "Objective",
            "Methodology",
            "Dataset",
            "Results",
            "Limitations",
            "Future Work"
        ]:

            value = analysis.get(field, "")

            if value and str(value).strip():

                story.append(
                    Paragraph(
                        f"<b>{escape(field)}:</b> "
                        f"{escape(str(value))}",
                        body_style
                    )
                )

                story.append(Spacer(1, 6))

    # Similarity comparison
    story.append(
        Paragraph("2. Paper Similarity Comparison", heading_style)
    )

    if comparison_result:

        names = comparison_result["names"]
        similarity = comparison_result["similarity"]

        table_data = [
            ["Research Paper"] + names
        ]

        for i, name in enumerate(names):

            row = [name]

            for j in range(len(names)):

                row.append(
                    f"{similarity[i][j] * 100:.2f}%"
                )

            table_data.append(row)

        comparison_table = Table(
            table_data,
            repeatRows=1
        )

        comparison_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ])
        )

        story.append(comparison_table)

    else:

        add_paragraph("Paper comparison has not been performed.")

    # Research gaps
    story.append(
        Paragraph("3. Potential Research Gaps", heading_style)
    )

    if gap_results:

        identified_gaps = gap_results.get(
            "identified_gaps", []
        )

        if identified_gaps:

            for index, gap in enumerate(
                identified_gaps,
                start=1
            ):

                add_paragraph(f"Potential Gap {index}")

                add_paragraph(
                    f"Text similarity: {gap['similarity']}%"
                )

                add_paragraph(
                    f"Paper 1: {gap['paper1']}"
                )

                add_paragraph(
                    f"Evidence: {gap['evidence1']}"
                )

                add_paragraph(
                    f"Paper 2: {gap['paper2']}"
                )

                add_paragraph(
                    f"Evidence: {gap['evidence2']}"
                )

        else:

            add_paragraph(
                "No cross-paper limitation matches were detected."
            )

    else:

        add_paragraph(
            "Research gap detection has not been performed."
        )

    # Recommendations
    story.append(
        Paragraph("4. Future Research Recommendations", heading_style)
    )

    if gap_results:

        if identified_gaps:

            for index, gap in enumerate(
                identified_gaps,
                start=1
            ):

                add_paragraph(
                    f"Recommendation {index}: Investigate the shared "
                    f"limitations identified in {gap['paper1']} and "
                    f"{gap['paper2']}. Explore improved methods, "
                    f"additional datasets, or broader evaluation."
                )

        else:

            add_paragraph(
                "Review the individual paper future-work sections "
                "to identify possible research directions."
            )

    else:

        add_paragraph(
            "Recommendations are unavailable because research "
            "gap detection has not yet been performed."
        )

    # Disclaimer
    story.append(Spacer(1, 15))

    story.append(
        Paragraph("Important Note", heading_style)
    )

    add_paragraph(
        "Research gaps and recommendations in this report are "
        "potential findings based on the uploaded papers. Textual "
        "similarity does not prove that a research topic is unexplored. "
        "Further literature review and expert validation are required."
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()
