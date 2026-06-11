from fastapi import FastAPI, UploadFile
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import uuid
import os
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from openai import OpenAI

load_dotenv()

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


def rewrite_query(question: str, client: OpenAI) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question as a declarative statement that would "
                    "appear in a document. Output only the rewritten statement, nothing else."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return res.choices[0].message.content.strip()


SCORE_THRESHOLD = 0.4


@app.post("/query")
def query_pdf(payload: dict):
    question = payload["question"]
    top_k = 5

    client = OpenAI()
    rewritten = rewrite_query(question, client)

    query_vec = embed_texts([rewritten])[0]
    store = QdrantStorage()
    found = store.search_vectors(query_vec, top_k)

    contexts = [c for c, s in zip(found["contexts"], found["scores"]) if s >= SCORE_THRESHOLD]

    if not contexts:
        return {"answer": "I couldn't find relevant information in the uploaded documents."}

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
    return {"answer": answer}
