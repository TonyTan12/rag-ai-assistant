import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI

PATHS = [
 r"D:\tonyt\OpenAI Retrieval Augmented Generation\Data\chroma_db",
 r"D:\tonyt\OpenAI Retrieval Augmented Generation\Rag-Assistant\Data\chroma_db",
]
COL = os.getenv("RAG_COLLECTION_NAME", "docs")

client = OpenAI()
emb = client.embeddings.create(model="text-embedding-3-small", input="hello").data[0].embedding
print("query embedding dim =", len(emb))

for p in PATHS:
    print("\n===", p)
    c = chromadb.PersistentClient(path=p, settings=Settings(anonymized_telemetry=False))
    try:
        col = c.get_collection(COL)
    except Exception as e:
        print("get_collection error:", e)
        continue
    try:
        print("count =", col.count())
        res = col.query(query_embeddings=[emb], n_results=1)
        print("query ok ids =", res.get("ids"))
    except Exception as e:
        print("query error:", e)
