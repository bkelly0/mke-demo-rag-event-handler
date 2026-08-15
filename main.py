import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Literal
from google import genai
from google.cloud import storage, bigquery
from google.genai import types
import vertexai
from vertexai.language_models import TextEmbeddingModel
import pypdf


app = FastAPI()
storage_client = storage.Client(project="bkelly-portfolio")

vertexai.init(
    project="bkelly-portfolio",
    location="us-central1",
)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
genai_client = genai.Client(
    vertexai=True,
    project="bkelly-portfolio",
    location="us-central1",
)

class StorageObjectData(BaseModel):
    bucket: str
    name: str
    generation: str
    metageneration: str
    size: str
    contentType: str | None = None
    crc32c: str | None = None
    md5Hash: str | None = None
    etag: str | None = None
    storageClass: str | None = None
    timeCreated: datetime | None = None
    updated: datetime | None = None


class GcpStorageFinalizedEvent(BaseModel):
    specversion: Literal["1.0"]
    id: str
    source: str
    type: Literal["google.cloud.storage.object.v1.finalized"]
    subject: str | None = None
    time: datetime
    datacontenttype: str | None = None
    data: StorageObjectData


def embed_text_chunks(chunks: list[dict], batch_size: int) -> list[list[float]]:
    all_embeddings = []
    
    # Process in batches to stay within API request limits
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        response = genai_client.models.embed_content(
            model="text-embedding-005",
            contents=texts,
            # Task type helps optimize vector clustering for semantic retrieval
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                title="PDF Document Chunks"
            )
        )
        
        # Extract vector lists
        for embedding in response.embeddings:
            all_embeddings.append(embedding.values)

    return all_embeddings


@app.post("/")
async def root(event: GcpStorageFinalizedEvent):
    bucket = event.data.bucket
    file_name = event.data.name

    if not file_name or not file_name.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    local_path = f"./tmp/{os.path.basename(file_name)}"
    bucket = storage_client.bucket(bucket)
    bucket.blob(file_name).download_to_filename(local_path)

    reader = pypdf.PdfReader(local_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append({"page": i + 1, "text": text.strip()})

        if not chunks:
            return {
                "message" : "No text found."
            }

    for i, chunk in enumerate(chunks):
        print("-------\n")
        print("test: " + str(chunk) + "\n")

    return {
        "message": "Downloaded file",
    }
    
#https://storage.googleapis.com/bkelly-mke-rag-data/CH295-sub1.pdf