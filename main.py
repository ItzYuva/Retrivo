import logging
from fastapi import FastAPI, UploadFile
from pathlib import Path
import tempfile
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from openai import OpenAI
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="retrivo-app",
    logger = logging.getLogger("uvicorn"),
    is_production = os.getenv("INNGEST_ENV", "").lower() == "production",
    serializer = inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id = "RAG: Ingest Pdf",
    trigger = inngest.TriggerEvent(event = "rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source = ctx.event.data.get("source", pdf_path)   # <-- changed
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source=source) 

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source = chunks_and_src.source                  # <-- changed
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert_vector(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id = "RAG: Query PDF",
    trigger = inngest.TriggerEvent(event = "rag/query_pdf_ai")
)

async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search_vectors(query_vec, top_k)
        return RAGSearchResult(contexts = found["contexts"], sources = found["sources"])
    
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed_and_search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely based on the context provided."
    )

    def _ask_llm() -> RAGQueryResult:
        client = OpenAI()
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        )
        answer = res.choices[0].message.content.strip()
        return RAGQueryResult(answer=answer, sources=found.sources, num_contexts=len(found.contexts))

    result = await ctx.step.run("llm-answer", _ask_llm, output_type=RAGQueryResult)
    return result.model_dump()

app = FastAPI()


@app.post("/ingest")
async def ingest_pdf(file: UploadFile):
    uploads_dir = Path(os.getenv("RETRIVO_UPLOAD_DIR", Path(tempfile.gettempdir()) / "retrivo_uploads"))
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.filename
    file_path.write_bytes(await file.read())

    source = file.filename
    chunks = load_and_chunk_pdf(str(file_path))
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source, "text": chunks[i]} for i in range(len(chunks))]
    QdrantStorage().upsert_vector(ids, vecs, payloads)
    return {"status": "ingested", "filename": source, "chunks": len(chunks)}


@app.post("/query")
def query_pdf(payload: dict):
    question = payload["question"]
    top_k = int(payload.get("top_k", 5))

    query_vec = embed_texts([question])[0]
    store = QdrantStorage()
    found = store.search_vectors(query_vec, top_k)

    contexts = found["contexts"]
    sources = found["sources"]

    context_block = "\n\n".join(f"- {c}" for c in contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely based on the context provided."
    )

    client = OpenAI()
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You answer questions using only the provided context."},
            {"role": "user", "content": user_content}
        ]
    )
    answer = res.choices[0].message.content.strip()
    return {"answer": answer, "sources": sources, "num_contexts": len(contexts)}


inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])