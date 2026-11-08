
import streamlit as st
import pandas as pd
import plotly.express as px
import pymupdf
import re

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.pdf_extractor import extract_text_from_pdf
from modules.paper_analyzer import analyze_paper
from modules.paper_comparator import compare_papers
from modules.pdf_report import generate_research_report


# ==================================================
# 1. PERMANENT PDF STORAGE
# ==================================================

STORAGE_DIR = Path("data") / "uploaded_papers"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_saved_papers():

    saved_papers = []

    for pdf_path in STORAGE_DIR.rglob("*.pdf"):

        try:

            with pymupdf.open(str(pdf_path)) as document:

                extracted_text = ""

                for page in document:
                    extracted_text += page.get_text() + "\n"

            if extracted_text.strip():

                saved_papers.append({
                    "name": pdf_path.name,
                    "text": extracted_text,
                    "area": pdf_path.parent.name,
                    "file_path": str(pdf_path)
                })

        except Exception as error:

            print(
                f"Could not load {pdf_path.name}: {error}"
            )

    return saved_papers


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 3. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1020, #111a33);
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: #ffffff;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #312e81, #1d4ed8);
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
}

.hero p {
    color: #e0e7ff;
    font-size: 17px;
}

.section-card {
    background-color: #17233f;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #334155;
    margin-bottom: 15px;
}

div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #2563eb);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 9px 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5, #1d4ed8);
    color: white;
}

h1, h2, h3 {
    color: #e0e7ff;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. SESSION STATE
# ==================================================

if "results" not in st.session_state:

    st.session_state.results = load_saved_papers()


if "comparison_result" not in st.session_state:

    st.session_state.comparison_result = None


if "gap_results" not in st.session_state:

    st.session_state.gap_results = None


if "generated_report" not in st.session_state:

    st.session_state.generated_report = None


# ==================================================
# 5. HELPER FUNCTIONS
# ==================================================

def get_current_area_papers():

    return [
        paper
        for paper in st.session_state.results
        if paper.get("area") == research_area
    ]


def is_available_content(content):

    if not content:
        return False

    unavailable = [
        "Not explicitly identified in the paper.",
        "Section heading found, but content was not extracted.",
        "Abstract not identified"
    ]

    return str(content).strip() not in unavailable


def show_hero(title, subtitle):

    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 6. SIDEBAR
# ==================================================

st.sidebar.title("🔬 ResearchLens AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📄 Upload Papers",
        "📊 Paper Analysis",
        "📚 Paper Comparison",
        "🔍 Research Gaps",
        "💡 Recommendations"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Research Area")

research_area = st.sidebar.selectbox(
    "Select your research domain",
    [
        "NLP and Generative AI",
        "AI and Robotics",
        "Cybersecurity",
        "Cloud Computing",
        "Computer Vision"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload research papers, analyze their content, "
    "compare their similarities, and explore potential "
    "research gaps."
)


# ==================================================
# 7. DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    show_hero(
        "🔬 ResearchLens AI",
        "AI-Powered Research Paper Comparison and Research Gap Detection System Using NLP"
    )

    papers = get_current_area_papers()

    total_papers = len(papers)

    total_words = sum(
        len(paper["text"].split())
        for paper in papers
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Papers Uploaded",
            total_papers
        )

    with col2:

        st.metric(
            "📝 Total Words",
            f"{total_words:,}"
        )

    with col3:

        st.metric(
            "🔬 Research Area",
            research_area
        )

    st.markdown("---")

    st.subheader("Welcome to ResearchLens AI")

    st.write(
        "ResearchLens AI helps researchers understand and compare "
        "multiple research papers using Natural Language Processing."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="section-card">
            <h3>📄 Upload Research Papers</h3>
            <p>Upload multiple research paper PDFs for analysis.</p>
        </div>

        <div class="section-card">
            <h3>📊 Paper Analysis</h3>
            <p>Extract objectives, methodology, datasets, results and limitations.</p>
        </div>

        <div class="section-card">
            <h3>📚 Paper Comparison</h3>
            <p>Compare papers using TF-IDF and cosine similarity.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="section-card">
            <h3>🔍 Research Gap Detection</h3>
            <p>Explore possible gaps based on limitations and future work.</p>
        </div>

        <div class="section-card">
            <h3>💡 Future Recommendations</h3>
            <p>Explore possible directions for future research.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Current Research Area")

    st.info(
        f"You are currently working in: {research_area}"
    )

    if papers:

        st.subheader("Saved Papers")

        for paper in papers:

            st.write(f"📄 {paper['name']}")

    else:

        st.warning(
            "No papers saved in this research area yet. "
            "Go to Upload Papers to begin."
        )

    # ==================================================
    # DOWNLOAD RESEARCH REPORT
    # ==================================================

    st.markdown("---")

    st.subheader("📥 Download Research Report")

    dashboard_papers = get_current_area_papers()

    if dashboard_papers:

        if st.button("Generate Research Report"):

            try:

                report_bytes = generate_research_report(
                    papers=dashboard_papers,
                    research_area=research_area,
                    comparison_result=st.session_state.comparison_result,
                    gap_results=st.session_state.gap_results
                )

                st.session_state.generated_report = report_bytes

                st.success(
                    "Research report generated successfully!"
                )

            except Exception as error:

                st.error(
                    f"Report generation failed: {error}"
                )

        if st.session_state.get("generated_report"):

            st.download_button(
                label="📄 Download Research Report (PDF)",
                data=st.session_state.generated_report,
                file_name="ResearchLens_AI_Report.pdf",
                mime="application/pdf"
            )

    else:

        st.info(
            "Upload research papers to generate a report."
        )


# ==================================================
# 8. UPLOAD PAPERS WITH PERMANENT STORAGE
# ==================================================

elif page == "📄 Upload Papers":

    show_hero(
        "📄 Upload Research Papers",
        "Upload and permanently store multiple research paper PDFs"
    )

    st.subheader("Selected Research Area")

    st.info(research_area)

    uploaded_files = st.file_uploader(
        "Choose your research paper PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="paper_uploader"
    )

    if uploaded_files:

        if st.button("🚀 Upload and Save Papers"):

            area_folder = STORAGE_DIR / research_area

            area_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            existing_names = {
                (paper["name"], paper.get("area"))
                for paper in st.session_state.results
            }

            success_count = 0

            progress = st.progress(0)

            status = st.empty()

            for index, file in enumerate(uploaded_files):

                safe_filename = Path(file.name).name

                paper_key = (
                    safe_filename,
                    research_area
                )

                save_path = area_folder / safe_filename

                if (
                    paper_key in existing_names
                    or save_path.exists()
                ):

                    status.warning(
                        f"{safe_filename} is already saved."
                    )

                else:

                    try:

                        status.info(
                            f"Processing {safe_filename}..."
                        )

                        extracted_text = extract_text_from_pdf(file)

                        if not extracted_text.strip():

                            st.error(
                                f"No readable text found in {safe_filename}. "
                                "Scanned PDFs are not supported yet."
                            )

                        else:

                            save_path.write_bytes(
                                file.getvalue()
                            )

                            st.session_state.results.append({

                                "name": safe_filename,

                                "text": extracted_text,

                                "area": research_area,

                                "file_path": str(save_path)

                            })

                            existing_names.add(paper_key)

                            success_count += 1

                            st.success(
                                f"✅ {safe_filename} saved successfully!"
                            )

                    except Exception as error:

                        st.error(
                            f"Could not save {safe_filename}: {error}"
                        )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.session_state.comparison_result = None

            st.session_state.gap_results = None

            st.session_state.generated_report = None

            status.empty()

            st.success(
                f"Completed! {success_count} new PDF(s) saved."
            )

    st.markdown("---")

    st.subheader("📂 Saved Research Papers")

    papers = get_current_area_papers()

    if papers:

        for index, paper in enumerate(papers, start=1):

            st.write(
                f"**{index}.** 📄 {paper['name']}"
            )

            st.caption(
                f"Words: {len(paper['text'].split()):,}"
            )

            st.caption(
                f"Storage location: {paper.get('file_path', 'Not available')}"
            )

    else:

        st.info(
            "No saved papers in this research area yet."
        )


# ==================================================
# 9. PAPER ANALYSIS
# ==================================================

elif page == "📊 Paper Analysis":

    show_hero(
        "📊 Research Paper Analysis",
        "Extract important information from uploaded research papers"
    )

    papers = get_current_area_papers()

    if not papers:

        st.warning(
            "Please upload research papers in this research area first."
        )

    else:

        selected_paper_name = st.selectbox(
            "Select a research paper",
            [paper["name"] for paper in papers]
        )

        selected_paper = next(
            paper for paper in papers
            if paper["name"] == selected_paper_name
        )

        analysis = analyze_paper(
            selected_paper["text"]
        )

        st.markdown("---")

        st.subheader(
            f"📄 {selected_paper_name}"
        )

        st.metric(
            "Total Words",
            f"{len(selected_paper['text'].split()):,}"
        )

        sections = [
            ("Title", "📌 Title"),
            ("Abstract", "📖 Abstract"),
            ("Objective", "🎯 Research Objective"),
            ("Methodology", "⚙️ Methodology"),
            ("Dataset", "📂 Dataset"),
            ("Results", "📈 Results"),
            ("Limitations", "⚠️ Limitations"),
            ("Future Work", "🚀 Future Work")
        ]

        for key, heading in sections:

            with st.expander(
                heading,
                expanded=(key == "Title")
            ):

                content = analysis.get(
                    key,
                    "Not explicitly identified in the paper."
                )

                if is_available_content(content):

                    st.write(content)

                else:

                    st.info(
                        "This information could not be clearly extracted "
                        "from the uploaded paper."
                    )

        with st.expander("📃 View Extracted PDF Text"):

            st.text_area(
                "Extracted text",
                selected_paper["text"],
                height=350
            )


# ==================================================
# 10. PAPER COMPARISON
# ==================================================

elif page == "📚 Paper Comparison":

    show_hero(
        "📚 Research Paper Comparison",
        "Compare research papers using NLP and cosine similarity"
    )

    papers = get_current_area_papers()

    if len(papers) < 2:

        st.warning(
            "Please upload at least two research papers "
            "in the same research area."
        )

    else:

        st.subheader("Select Papers to Compare")

        selected_names = st.multiselect(
            "Choose two or more papers",
            options=[
                paper["name"]
                for paper in papers
            ],
            default=[
                paper["name"]
                for paper in papers
            ]
        )

        if st.button("🔍 Compare Research Papers"):

            if len(selected_names) < 2:

                st.warning(
                    "Please select at least two papers."
                )

            else:

                selected_papers = [
                    paper
                    for paper in papers
                    if paper["name"] in selected_names
                ]

                comparison, error = compare_papers(
                    selected_papers
                )

                if error:

                    st.error(error)

                else:

                    st.session_state.comparison_result = {

                        "names": comparison["names"],

                        "similarity": comparison["similarity"]

                    }

                    st.session_state.generated_report = None

        result = st.session_state.comparison_result

        if result:

            names = result["names"]

            matrix = result["similarity"]

            st.markdown("---")

            st.subheader("📊 Similarity Matrix")

            similarity_df = pd.DataFrame(
                matrix * 100,
                index=names,
                columns=names
            )

            similarity_df = similarity_df.round(2)

            st.dataframe(
                similarity_df.style.format("{:.2f}%"),
                use_container_width=True
            )

            st.markdown("---")

            st.subheader("📈 Paper Similarity Chart")

            pair_labels = []

            pair_scores = []

            for i in range(len(names)):

                for j in range(i + 1, len(names)):

                    pair_labels.append(
                        f"{names[i]} vs {names[j]}"
                    )

                    pair_scores.append(
                        round(
                            float(matrix[i][j]) * 100,
                            2
                        )
                    )

            if pair_labels:

                chart_df = pd.DataFrame({

                    "Paper Pair": pair_labels,

                    "Similarity (%)": pair_scores

                })

                fig = px.bar(
                    chart_df,
                    x="Similarity (%)",
                    y="Paper Pair",
                    orientation="h",
                    text="Similarity (%)",
                    range_x=[0, 100],
                    title="Research Paper Similarity"
                )

                fig.update_traces(
                    texttemplate="%{text:.2f}%",
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_layout(

                    xaxis_title="Similarity (%)",

                    yaxis_title="Paper Comparison",

                    height=max(
                        350,
                        len(pair_labels) * 75
                    ),

                    margin=dict(r=80)

                )

                fig.update_xaxes(
                    range=[0, 100],
                    ticksuffix="%",
                    dtick=10
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.markdown("---")

            st.subheader("📋 Comparison Summary")

            for i in range(len(names)):

                for j in range(i + 1, len(names)):

                    score = float(matrix[i][j]) * 100

                    st.write(
                        f"**{names[i]}** vs **{names[j]}**: "
                        f"{score:.2f}% similarity"
                    )

            st.caption(
                "Similarity is calculated using TF-IDF and cosine similarity. "
                "It measures textual similarity, not research quality."
            )


# ==================================================
# 11. RESEARCH GAP DETECTION
# ==================================================

elif page == "🔍 Research Gaps":

    show_hero(
        "🔍 Research Gap Detection",
        "Compare limitations across papers and identify potential research gaps"
    )

    papers = get_current_area_papers()

    if len(papers) < 2:

        st.warning(
            "Upload at least two research papers in this research area."
        )

    else:

        st.write(
            "This module compares limitations from multiple research "
            "papers using TF-IDF and cosine similarity."
        )

        if st.button("🔎 Detect Potential Research Gaps"):

            gap_results = []

            for paper in papers:

                analysis = analyze_paper(paper["text"])

                limitations = analysis.get("Limitations", "")
                future_work = analysis.get("Future Work", "")

                gap_results.append({
                    "name": paper["name"],
                    "limitations": limitations,
                    "future_work": future_work
                })

            limitation_sentences = []

            for item in gap_results:

                if is_available_content(item["limitations"]):

                    sentences = re.split(
                        r'(?<=[.!?])\s+',
                        item["limitations"]
                    )

                    for sentence in sentences:

                        sentence = sentence.strip()

                        if len(sentence.split()) >= 5:

                            limitation_sentences.append({
                                "paper": item["name"],
                                "text": sentence
                            })

            identified_gaps = []

            if len(limitation_sentences) >= 2:

                documents = [
                    item["text"]
                    for item in limitation_sentences
                ]

                try:

                    vectorizer = TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2)
                    )

                    matrix = vectorizer.fit_transform(documents)

                    similarity = cosine_similarity(matrix)

                    for i in range(len(limitation_sentences)):

                        for j in range(i + 1, len(limitation_sentences)):

                            first = limitation_sentences[i]
                            second = limitation_sentences[j]

                            score = similarity[i][j] * 100

                            if (
                                first["paper"] != second["paper"]
                                and score >= 15
                            ):

                                identified_gaps.append({
                                    "paper1": first["paper"],
                                    "paper2": second["paper"],
                                    "evidence1": first["text"],
                                    "evidence2": second["text"],
                                    "similarity": round(score, 2)
                                })

                except ValueError:

                    identified_gaps = []

            st.session_state.gap_results = {
                "papers": gap_results,
                "identified_gaps": identified_gaps
            }

            st.session_state.generated_report = None

        gap_data = st.session_state.gap_results

        if gap_data:

            st.markdown("---")

            st.subheader("📊 Research Gap Analysis")

            paper_results = gap_data["papers"]
            identified_gaps = gap_data["identified_gaps"]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Papers Analyzed",
                    len(paper_results)
                )

            with col2:

                st.metric(
                    "Papers with Limitations",
                    sum(
                        is_available_content(p["limitations"])
                        for p in paper_results
                    )
                )

            with col3:

                st.metric(
                    "Potential Gap Matches",
                    len(identified_gaps)
                )

            st.markdown("---")

            st.subheader("🔍 Potential Research Gaps")

            if identified_gaps:

                for index, gap in enumerate(
                    identified_gaps,
                    start=1
                ):

                    with st.expander(
                        f"Potential Gap {index} — "
                        f"{gap['similarity']}% text similarity",
                        expanded=True
                    ):

                        st.write(
                            f"**Paper 1:** {gap['paper1']}"
                        )

                        st.info(gap["evidence1"])

                        st.write(
                            f"**Paper 2:** {gap['paper2']}"
                        )

                        st.info(gap["evidence2"])

                        st.write(
                            "**Interpretation:** These papers contain "
                            "similar limitation statements. This may "
                            "indicate an area that deserves further study."
                        )

            else:

                st.info(
                    "No sufficiently similar limitation sentences "
                    "were identified across different papers. "
                    "This does not mean there are no research gaps."
                )

            st.markdown("---")

            st.subheader("📄 Individual Paper Findings")

            for item in paper_results:

                with st.expander(item["name"]):

                    st.markdown("**⚠️ Limitations**")

                    if is_available_content(item["limitations"]):
                        st.write(item["limitations"])
                    else:
                        st.info("No clear limitations extracted.")

                    st.markdown("**🚀 Future Work**")

                    if is_available_content(item["future_work"]):
                        st.write(item["future_work"])
                    else:
                        st.info("No clear future work extracted.")

            st.warning(
                "Important: Similarity between limitation statements "
                "is only a signal for a potential research gap. "
                "It does not prove that the topic is unexplored. "
                "Review the original papers before accepting a gap."
            )


# ==================================================
# 12. FUTURE RESEARCH RECOMMENDATIONS
# ==================================================

elif page == "💡 Recommendations":

    show_hero(
        "💡 Future Research Recommendations",
        "Explore future research directions based on detected research gaps"
    )

    papers = get_current_area_papers()

    if not papers:

        st.warning("Please upload research papers first.")

    else:

        st.write(
            "Recommendations are derived from the limitations "
            "and future-work statements in your uploaded papers."
        )

        gap_data = st.session_state.gap_results

        if not isinstance(gap_data, dict):

            st.warning(
                "Please visit Research Gaps and click "
                "'Detect Potential Research Gaps' first."
            )

            st.info(
                "Recommendations will be generated using the "
                "research gaps detected in that section."
            )

        else:

            identified_gaps = gap_data.get(
                "identified_gaps", []
            )

            paper_results = gap_data.get(
                "papers", []
            )

            st.markdown("---")

            st.subheader("📊 Recommendation Overview")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Papers Analyzed",
                    len(paper_results)
                )

            with col2:

                st.metric(
                    "Potential Research Gaps",
                    len(identified_gaps)
                )

            st.markdown("---")

            st.subheader("🚀 Suggested Future Research Directions")

            if identified_gaps:

                for index, gap in enumerate(
                    identified_gaps,
                    start=1
                ):

                    with st.expander(
                        f"Research Recommendation {index}",
                        expanded=True
                    ):

                        st.markdown(
                            "### 🔍 Identified Research Problem"
                        )

                        st.write(
                            "Similar limitation statements were "
                            "identified in two different papers."
                        )

                        st.markdown("**Supporting Paper 1**")

                        st.write(gap["paper1"])

                        st.info(gap["evidence1"])

                        st.markdown("**Supporting Paper 2**")

                        st.write(gap["paper2"])

                        st.info(gap["evidence2"])

                        st.markdown(
                            "### 💡 Future Research Recommendation"
                        )

                        st.write(
                            "Investigate this shared limitation "
                            "by exploring improved methods, "
                            "additional datasets, or broader "
                            "experimental evaluation."
                        )

                        st.markdown(
                            "### 🎯 Why This Research Is Important"
                        )

                        st.write(
                            "Addressing this limitation may improve "
                            "the reliability, performance, or "
                            "practical applicability of existing "
                            "research approaches."
                        )

                        st.caption(
                            "Evidence-based suggestion. "
                            "Further literature review is required "
                            "before treating it as a novel research topic."
                        )

            else:

                st.warning(
                    "No cross-paper limitation matches were detected. "
                    "Please review individual paper future-work sections "
                    "below for possible research directions."
                )

            st.markdown("---")

            st.subheader("📚 Individual Paper Research Directions")

            for paper in paper_results:

                with st.expander(paper["name"]):

                    st.markdown("**Future Work**")

                    if is_available_content(paper["future_work"]):

                        st.write(paper["future_work"])

                    else:

                        st.info(
                            "No future-work section was extracted."
                        )

                    st.markdown("**Limitations**")

                    if is_available_content(paper["limitations"]):

                        st.write(paper["limitations"])

                    else:

                        st.info(
                            "No limitations section was extracted."
                        )

            st.markdown("---")

            st.info(
                "Note: These recommendations are rule-based suggestions "
                "derived from extracted research evidence. They are not "
                "AI-generated scientific conclusions or proof of novelty."
            )


# ==================================================
# 13. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8;">
        <p>🔬 ResearchLens AI</p>
        <p>
            AI-Powered Research Paper Comparison and
            Research Gap Detection System Using NLP
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
