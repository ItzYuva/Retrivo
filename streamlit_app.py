# streamlit_app.py
import asyncio
import threading
import queue
import time
from pathlib import Path
from tempfile import gettempdir

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# -----------------------
# Helpers
# -----------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    """
    Return a cached inngest client. Adjust constructor args if your installed
    inngest client requires api_base or api_key.
    """
    api_base = os.getenv("INNGEST_API_BASE")
    api_key = os.getenv("INNGEST_API_KEY")
    # Create client in a way that works for local dev and cloud if env is set.
    if api_base or api_key:
        return inngest.Inngest(app_id="rag_app", is_production=False, api_base=api_base, api_key=api_key)
    return inngest.Inngest(app_id="rag_app", is_production=False)


def run_coro_in_thread(coro):
    """
    Run coroutine in a new event loop inside a background thread and return result.
    This avoids asyncio.run inside an already-running loop (Streamlit).
    """
    q = queue.Queue()

    def _target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(coro)
            q.put((True, res))
        except Exception as e:
            q.put((False, e))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    ok, res = q.get()
    if not ok:
        raise res
    return res


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path(os.getenv("RETRIVO_UPLOAD_DIR", Path(gettempdir()) / "retrivo_uploads"))
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> str:
    """
    Send the ingest event to Inngest. Returns event id (string) where possible.
    """
    client = get_inngest_client()
    payload = {
        "pdf_path": str(pdf_path.resolve()),
        "source": pdf_path.name,
        "source_id": pdf_path.name,
    }
    res = await client.send(inngest.Event(name="rag/ingest_pdf", data=payload))

    # Normalize return to an event id string where possible
    if isinstance(res, dict):
        return res.get("id") or res.get("event_id") or str(res)
    if isinstance(res, (list, tuple)) and res:
        # sometimes the client returns [event_id, ...] or similar
        return res[0]
    return str(res)


async def send_rag_query_event(question: str, top_k: int) -> str:
    """
    Send a query event that triggers the query function. Returns event id.
    """
    client = get_inngest_client()
    payload = {"question": question, "top_k": top_k}
    res = await client.send(inngest.Event(name="rag/query_pdf_ai", data=payload))

    if isinstance(res, dict):
        return res.get("id") or res.get("event_id") or str(res)
    if isinstance(res, (list, tuple)) and res:
        return res[0]
    return str(res)


# -----------------------
# Polling / API helpers
# -----------------------
def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    try:
        resp = requests.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", []) or []
    except requests.HTTPError:
        return []
    except requests.RequestException:
        return []


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = (run.get("status") or "").lower()
            last_status = status or last_status
            if status in ("completed", "succeeded", "success", "finished"):
                return run.get("output") or {}
            if status in ("failed", "cancelled", "canceled"):
                # include run debug info when possible
                raise RuntimeError(f"Function run {status}. run: {run}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


# -----------------------
# Streamlit UI
# -----------------------
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        try:
            event_id = run_coro_in_thread(send_rag_ingest_event(path))
            # Small pause for UX continuity
            time.sleep(0.3)
            st.success(f"Triggered ingestion for: {path.name} (event {event_id})")
        except Exception as e:
            st.error(f"Failed to trigger ingest: {e}")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            try:
                event_id = run_coro_in_thread(send_rag_query_event(question.strip(), int(top_k)))

                # Normalize event_id if library returned dict/list
                if isinstance(event_id, dict):
                    event_id = event_id.get("id") or event_id.get("event_id") or str(event_id)
                if isinstance(event_id, (list, tuple)) and event_id:
                    event_id = event_id[0]

                output = wait_for_run_output(event_id)
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
            except TimeoutError as te:
                st.error(f"Timed out waiting for an answer: {te}")
            except RuntimeError as re:
                st.error(f"Query failed: {re}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
