import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url=None, collection_name="docs", dim=3072):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30, port=443)
        self.collection = collection_name
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        from qdrant_client.models import PayloadSchemaType
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def upsert_vector(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def source_exists(self, source: str) -> bool:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0

    def search_vectors(self, query_vector, top_k: int=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        scores = []

        for r in results.points:
            payload = getattr(r, 'payload', None) or {}
            text = payload.get('text', '')
            if text:
                contexts.append(text)
                scores.append(r.score)

        return {"contexts": contexts, "scores": scores}