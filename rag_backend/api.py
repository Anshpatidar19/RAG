"""
FastAPI entrypoint. Wraps the KnowledgeBase facade in REST endpoints
for the React frontend to call. Run with:
    uvicorn api:app --reload
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embedding.embedder import Embedder
from generation.generator import AnswerGenerator
from ingestion.pipeline import IngestionPipeline
from kb.knowledge_base import KnowledgeBase
from retrieval.retriever import Retriever
from storage.document_repository import DocumentRepository
from storage.vector_store import VectorStoreRepository

app = FastAPI(title="Universal RAG API")

# Allow the Vite dev server to call this API. Add your deployed
# frontend's URL here too once you host it somewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Wiring: one shared KnowledgeBase instance for the whole app ---
_embedder = Embedder()
_vector_store = VectorStoreRepository()
_document_repository = DocumentRepository()
_pipeline = IngestionPipeline(_embedder, _vector_store)
_retriever = Retriever(_embedder, _vector_store)
_generator = AnswerGenerator()
kb = KnowledgeBase(_pipeline, _retriever, _generator, _vector_store, _document_repository)


# --- Request/response schemas ---
class AskRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    filename: str
    page_number: int | None
    score: float
    text_snippet: str


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    sources: list[SourceResponse]


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    num_chunks: int


# --- Helpers ---
def _to_document_response(doc) -> DocumentResponse:
    return DocumentResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        status=doc.status.value,
        num_chunks=doc.num_chunks,
    )


def _save_upload_to_temp(file: UploadFile) -> str:
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...)):
    tmp_path = _save_upload_to_temp(file)
    try:
        document = kb.add_document(tmp_path, display_filename=file.filename)
        return _to_document_response(document)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents():
    return [_to_document_response(d) for d in kb.list_documents()]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if kb.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    kb.delete_document(doc_id)
    return {"deleted": doc_id}


@app.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(doc_id: str, file: UploadFile = File(...)):
    if kb.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    tmp_path = _save_upload_to_temp(file)
    try:
        document = kb.update_document(doc_id, tmp_path, display_filename=file.filename)
        return _to_document_response(document)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    answer = kb.ask(request.question)
    sources = [
        SourceResponse(
            filename=r.source_filename,
            page_number=r.chunk.page_number,
            score=r.score,
            text_snippet=r.chunk.text[:200],
        )
        for r in answer.sources
    ]
    return AskResponse(answer=answer.text, answerable=answer.answerable, sources=sources)