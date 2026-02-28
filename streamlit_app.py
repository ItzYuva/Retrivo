import streamlit as st
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="Retrivo", page_icon="📄", layout="centered")

API_BASE = os.getenv("RETRIVO_API_BASE", "http://127.0.0.1:8000")

# -----------------------
# Streamlit UI
# -----------------------
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and ingesting PDF..."):
        try:
            resp = requests.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=300.0,
            )
            resp.raise_for_status()
            result = resp.json()
            st.success(f"Ingested {uploaded.name} ({result.get('chunks', '?')} chunks)")
        except Exception as e:
            st.error(f"Failed to ingest: {e}")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Generating answer..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question.strip(), "top_k": int(top_k)},
                    timeout=60.0,
                )
                resp.raise_for_status()
                output = resp.json()
                answer = output.get("answer", "")
                sources = output.get("sources", [])
                num_ctx = output.get("num_contexts", None)

                st.subheader("Answer")
                st.write(answer or "(No answer)")

                if num_ctx is not None:
                    st.caption(f"Number of contexts used: {num_ctx}")

                if sources:
                    st.caption("Sources")
                    for s in sources:
                        st.write(f"- {s}")
            except requests.RequestException as e:
                st.error(f"Query failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
