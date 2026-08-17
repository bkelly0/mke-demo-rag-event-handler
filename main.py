import os
import logging
import json
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google.cloud.logging_v2.handlers import StructuredLogHandler
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from datetime import datetime
from google import genai
from google.cloud import storage, bigquery
from google.genai import types
from google import genai
from google.genai import types
import pypdf
from dotenv import load_dotenv

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
BIGQUERY_DATASET = os.environ["BIGQUERY_DATASET"]
BIGQUERY_TABLE = os.environ["BIGQUERY_TABLE"]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

app = FastAPI()
storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=REGION,
)

#request objects for binary mode where metadata is provided in the headers
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
    data: StorageObjectData

def build_logger() -> logging.Logger:
    logger = logging.getLogger("rag-event-handler")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        # Structured JSON to stdout; Cloud Run/GCP ingests this automatically.
        handler = StructuredLogHandler()
        logger.addHandler(handler)

    return logger

logger = build_logger()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

def embed_text_chunks(chunks: list[dict]) -> list[list[float]]:
    all_embeddings = []
    batch_size = 250; # Vertex AI embedding API supports up to 250 texts per batch request
    
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

def download_file(event: GcpStorageFinalizedEvent) -> str:
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
    return local_path

def process_pdf_file(local_path: str) -> tuple[list[dict[str, str | int]], list[list[float]]]:
    reader = pypdf.PdfReader(local_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append({"page": i + 1, "text": text.strip()})

    embeddings = embed_text_chunks(chunks)
    return chunks, embeddings

def insert_into_bigquery(
    chunks: list[dict[str, str | int]],
    embeddings: list[list[float]],
    doc_id: str,
    source_type: str,
) -> None: 
        rows = []
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            rows.append(
                {
                    "chunk_id": doc_id + "_" + str(index),
                    "doc_id": doc_id,
                    "source_type": source_type,
                    "content": str(chunk["text"]),   
                    "metadata": json.dumps({"page": chunk["page"]}),
                    "embedding": vector
                }
            )
        
        table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
        errors = bq_client.insert_rows_json(table, rows)

        if errors:
            logger.error(str(errors))
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to insert rows into BigQuery",
                    "errors": errors,
                },
            )

@app.post("/")
async def root(event: GcpStorageFinalizedEvent, ce_type: str = Header(..., alias="ce-type")):
    if ce_type != "google.cloud.storage.object.v1.finalized":
        logger.error("Unsupported event type")
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported event type {ce_type}",
                "errors": [],
            },
        )

    try:
        local_file_path = download_file(event);
        chunks, embeddings = process_pdf_file(local_file_path);
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Chunks not found",
                    "errors": [],
                },
            )
        insert_into_bigquery(chunks, embeddings, event.data.name, "pdf")
    finally:
        os.remove(local_file_path)
    return {
        "message": "File Processed",
    }

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"422 Validation Error: {exc.errors()}")
    logger.info(f"Received Body: {await request.body()}")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )