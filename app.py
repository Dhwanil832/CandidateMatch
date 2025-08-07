import streamlit as st
import pandas as pd

from utils import reset_app
from resume_parser import parse_resume
from embedding_utils import get_embedding
from similarity import rank_candidates
from summary_generator import generate_summary_gpt35

# ----------------- SESSION STATE INIT -----------------
if "step" not in st.session_state:
    st.session_state.step = "input"

if "resumes" not in st.session_state:
    st.session_state.resumes = []  # list of dicts: filename, name, text, embedding

if "summaries" not in st.session_state:
    st.session_state.summaries = {}  # cache for GPT summaries

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="CandidateMatch.AI", layout="wide")
st.title("CandidateMatch.AI")
st.markdown("Match resumes to a job description with AI-powered ranking and summaries.")
st.write("---")

# ----------------- ANALYSIS FUNCTION -----------------
def run_analysis():
    st.subheader("Analyzing Resumes...")
    progress = st.progress(0)

    # Step 1: Generate job description embedding
    job_embedding = get_embedding(st.session_state.job_description)

    # Step 2: Generate resume embeddings
    for i, r in enumerate(st.session_state.resumes):
        if r['text'] is not None:
            r['embedding'] = get_embedding(r['text'])
        else:
            r['embedding'] = None
        progress.progress((i + 1) / len(st.session_state.resumes))

    # Step 3: Rank candidates
    ranking_df = rank_candidates(job_embedding, st.session_state.resumes)

    # Step 4: Add summaries + penalty adjustments
    summaries = []
    final_scores = []
    for _, row in ranking_df.iterrows():
        resume_text = next((r['text'] for r in st.session_state.resumes if r['filename'] == row['File Name']), None)
        summary, penalty = generate_summary_gpt35(
            st.session_state.job_description,
            resume_text,
            st.session_state.summaries
        )
        adjusted_score = round(row['Similarity Score'] * penalty, 2)
        summaries.append(summary)
        final_scores.append(adjusted_score)

    ranking_df['Similarity Score'] = final_scores
    ranking_df['Fit Summary'] = summaries

    # Save results
    st.session_state.result_df = ranking_df.sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)
    st.session_state.step = "results"
    st.rerun()

# ----------------- LAYOUT -----------------
col1, col2 = st.columns(2)

# ----------------- LEFT PANEL: JOB DESCRIPTION -----------------
with col1:
    st.subheader("Job Description")
    example_jd = "We are seeking a Machine Learning Engineer with experience in Python, NLP, and model deployment..."
    if st.button("Load Example Job Description"):
        st.session_state.job_description = example_jd

    job_description = st.text_area(
        "Paste Job Description Here",
        value=st.session_state.job_description,
        height=300
    )
    st.session_state.job_description = job_description

# ----------------- RIGHT PANEL: RESUME UPLOAD / RESULTS -----------------
with col2:
    if st.session_state.step == "input":
        st.subheader("Upload Resumes (PDF, up to 30 files)")
        uploaded_files = st.file_uploader(
            "Drag & drop resumes here",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if len(uploaded_files) > 30:
                st.error("You uploaded more than 30 resumes. Only the first 30 will be processed automatically.")
                uploaded_files = uploaded_files[:30]  # Truncate to first 30

            st.session_state.resumes = [parse_resume(f) for f in uploaded_files]
            st.write(f"{len(st.session_state.resumes)} resumes ready for analysis.")

            # Auto-run analysis if 30 resumes uploaded
            if len(st.session_state.resumes) == 30:
                run_analysis()
            elif st.button("Start Analysis"):
                run_analysis()

    elif st.session_state.step == "results":
        st.subheader("Ranked Candidates")
        st.dataframe(st.session_state.result_df, use_container_width=True)

        # Download CSV
        csv = st.session_state.result_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="candidate_ranking.csv",
            mime="text/csv"
        )

        # Reset button
        if st.button("Reset App"):
            reset_app()
