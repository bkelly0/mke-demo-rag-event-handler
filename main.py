import os
import logging
import json
import re
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google.cloud.logging_v2.handlers import StructuredLogHandler
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from datetime import datetime
from google.cloud import storage, bigquery
from google.api_core.exceptions import Forbidden, GoogleAPICallError, NotFound
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import pypdf
from dotenv import load_dotenv

COST_PER_1M_TOKENS = 0.10
MODEL = "text-embedding-005"

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
BIGQUERY_DATASET = os.environ["BIGQUERY_DATASET"]
BIGQUERY_TABLE = os.environ["BIGQUERY_TABLE"]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("true", "1") # if true, do not call model or insert into BigQuery

app = FastAPI()
storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)

vertexai.init(project=PROJECT_ID, location=REGION)
embedding_model = TextEmbeddingModel.from_pretrained(MODEL)

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

if DRY_RUN:
    logger.warning("RUNNING IN DRY_RUN MODE. No requests will be sent to the model and data is not persisted.")

def embed_text_chunks(chunks: list[dict]) -> list[list[float]]:
    all_embeddings = []
    all_estimated_costs = []
    
    # Process in batches to stay within API request limits
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        all_estimated_costs.append(estimate_embedding_cost(texts))

        if not DRY_RUN:
            logger.info("requesting embeddings...")
            response = embedding_model.get_embeddings(
                [
                    TextEmbeddingInput(
                        text=text,
                        task_type="RETRIEVAL_DOCUMENT",
                        title="PDF Document Chunks",
                    )
                    for text in texts
                ]
            )
            
            # Extract vector lists
            for embedding in response:
                all_embeddings.append(embedding.values)

    return all_embeddings, all_estimated_costs

def download_file(object_data: StorageObjectData) -> str:
    bucket = object_data.bucket
    file_name = object_data.name

    if not file_name or not file_name.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    local_path = f"./tmp/{os.path.basename(file_name)}"
    try:
        os.makedirs("./tmp", exist_ok=True)
        bucket = storage_client.bucket(object_data.bucket)
        bucket.blob(file_name).download_to_filename(local_path)
        return local_path
    except NotFound as ex:
        raise HTTPException(404, "The requested PDF was not found.") from ex
    except Forbidden as ex:
        raise HTTPException(403, "Permission denied when downloading the PDF.") from ex
    except GoogleAPICallError as ex:
        logger.exception("Google Cloud Storage download failed")
        raise HTTPException(502, "Cloud Storage download failed.") from ex
    except OSError as ex:
        logger.exception("Local file operation failed")
        raise HTTPException(500, "Unable to save the downloaded PDF.") from ex

def chunk_text_by_paragraphs(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
    if not paragraphs:
        return [cleaned[:chunk_size]]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if not current:
            current = paragraph
            continue

        if len(current) + 2 + len(paragraph) <= chunk_size:
            current = f"{current}\n\n{paragraph}"
            continue

        chunks.append(current)
        current = paragraph

    if current:
        chunks.append(current)

    # Fall back to a character-based split only for very large single paragraphs.
    refined_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            refined_chunks.append(chunk)
            continue

        for start in range(0, len(chunk), chunk_size):
            fragment = chunk[start:start + chunk_size].strip()
            if fragment:
                refined_chunks.append(fragment)

    return refined_chunks


def process_pdf_file(local_path: str, doc_name: str, event_id: str) -> tuple[list[dict[str, str | int]], list[list[float]]]:
    reader = pypdf.PdfReader(local_path)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text:
            continue

        for chunk_text in chunk_text_by_paragraphs(text, CHUNK_SIZE):
            chunks.append({
                "page": page_number,
                "text": chunk_text,
            })

    embeddings, estimated_costs = embed_text_chunks(chunks)
    total_cost = log_costs(estimated_costs, local_path)
    write_document_log_to_bigquery(doc_name, total_cost, event_id)
    
    return chunks, embeddings


def insert_into_bigquery(
    chunks: list[dict[str, str | int]],
    embeddings: list[list[float]],
    doc_id: str,
    source_type: str,
) -> None: 
        logger.info("inserting data...")
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
async def root(
    object_data: StorageObjectData,
    ce_type: str = Header(..., alias="ce-type"),
    ce_id: str = Header(..., alias="ce-id"),
):
    if ce_type != "google.cloud.storage.object.v1.finalized":
        logger.error("Unsupported event type")
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported event type {ce_type}",
                "errors": [],
            },
        )

    local_file_path = None
    try:
        local_file_path = download_file(object_data);
        chunks, embeddings = process_pdf_file(local_file_path, object_data.name, ce_id);
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Chunks not found",
                    "errors": [],
                },
            )
        if not DRY_RUN:
            insert_into_bigquery(chunks, embeddings, object_data.name, "pdf")
    finally:
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)
    return {
        "message": "File Processed",
    }

def estimate_embedding_cost(texts: list[str]) -> dict:
    # Avoid a separate API call; four characters is a reasonable rough token estimate.
    total_tokens = sum(max(1, (len(text) + 3) // 4) for text in texts)

    estimated_cost = (total_tokens / 1_000_000) * COST_PER_1M_TOKENS

    return {
        "item_count": len(texts),
        "total_tokens": total_tokens,
        "estimated_cost": f"${estimated_cost:.8f}"
    }

def log_costs(estimated_costs, file) -> float:
    num_requests = len(estimated_costs)
    total_cost = 0.0
    total_tokens = 0
    for estimate in estimated_costs:
        total_cost += float(estimate["estimated_cost"].removeprefix("$"))
        total_tokens += estimate["total_tokens"]

    logger.info(f"Estimation: {total_tokens} tokens across {num_requests} requests for file {file}, ${total_cost:.8f}")

    return total_cost


def write_document_log_to_bigquery(name: str, estimated_cost: float, event_id: str) -> None:
    logger.info("writing cost log...")
    row = {
        "name": name,
        # BigQuery NUMERIC supports at most 9 decimal digits of scale.
        "estimated_cost": round(estimated_cost, 9),
        "dry_run": DRY_RUN,
        "event_id": event_id,
    }

    table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.documents"
    errors = bq_client.insert_rows_json(table, [row])

    if errors:
        logger.error(str(errors))
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to insert cost log row into BigQuery",
                "errors": errors,
            },
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"422 Validation Error: {exc.errors()}")
    logger.info(f"Received Body: {await request.body()}")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )