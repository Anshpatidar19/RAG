"""
FastAPI entrypoint. Wraps the KnowledgeBase facade (for document CRUD)
and the Supervisor agent (for answering questions) in REST endpoints
for the React frontend to call. Run with:
    uvicorn api:app --reload
"""

import io
import mimetypes
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from gtts import gTTS
from google import genai
from pydantic import BaseModel

from agents.greeting import GreetingAgent
from agents.reflection import ReflectionAgent
from agents.supervision import Supervisor
from config import settings
from embedding.embedder import Embedder
from generation.generator import AnswerGenerator
from ingestion.pipeline import IngestionPipeline
from kb.knowledge_base import KnowledgeBase
from retrieval.retriever import Retriever
from retrieval.vector_search import VectorSearchTool
from retrieval.web_search import WebSearchTool
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

# --- Where original uploaded files are kept permanently, so they can be opened later ---
FILES_DIR = Path(__file__).parent / "uploaded_files"
FILES_DIR.mkdir(exist_ok=True)

# --- Wiring: document CRUD (unchanged) ---
_embedder = Embedder()
_vector_store = VectorStoreRepository()
_document_repository = DocumentRepository()
_pipeline = IngestionPipeline(_embedder, _vector_store)
_retriever = Retriever(_embedder, _vector_store)
_generator = AnswerGenerator()
kb = KnowledgeBase(_pipeline, _retriever, _generator, _vector_store, _document_repository)

# --- Wiring: the agent system, used for /ask ---
_gemini_client = genai.Client(api_key=settings.gemini_api_key)
_vector_search_tool = VectorSearchTool(_retriever)
_web_search_tool = WebSearchTool()
_greeting_agent = GreetingAgent(_gemini_client)
_reflection_agent = ReflectionAgent(_gemini_client)
supervisor = Supervisor(
    client=_gemini_client,
    vector_search=_vector_search_tool,
    web_search=_web_search_tool,
    greeting_agent=_greeting_agent,
    reflection_agent=_reflection_agent,
)


# --- Request/response schemas ---
class AskRequest(BaseModel):
    question: str


class TTSRequest(BaseModel):
    text: str


class SourceResponse(BaseModel):
    type: str  # "document" | "web"
    label: str  # filename for documents, page title for web
    url: str | None = None
    page_number: int | None = None
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
    file_url: str


# --- Helpers ---
def _to_document_response(doc) -> DocumentResponse:
    return DocumentResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        status=doc.status.value,
        num_chunks=doc.num_chunks,
        file_url=f"/documents/{doc.doc_id}/file",
    )


def _save_upload_to_temp(file: UploadFile) -> str:
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def _persist_file(tmp_path: str, doc_id: str, original_filename: str) -> str:
    """Move the temp upload into permanent storage, named by doc_id so it survives restarts."""
    suffix = Path(original_filename).suffix
    dest = FILES_DIR / f"{doc_id}{suffix}"
    # Clean up any previous file for this doc_id (relevant on update, if extension changed)
    for existing in FILES_DIR.glob(f"{doc_id}.*"):
        existing.unlink(missing_ok=True)
    shutil.move(tmp_path, dest)
    return str(dest)


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...)):
    tmp_path = _save_upload_to_temp(file)
    try:
        document = kb.add_document(tmp_path, display_filename=file.filename)
        permanent_path = _persist_file(tmp_path, document.doc_id, file.filename)
        document.filepath = permanent_path
        kb.document_repository.save(document)
        return _to_document_response(document)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents():
    return [_to_document_response(d) for d in kb.list_documents()]


@app.get("/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    document = kb.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    file_path = Path(document.filepath)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    media_type, _ = mimetypes.guess_type(document.filename)
    return FileResponse(
        file_path,
        filename=document.filename,
        media_type=media_type or "application/octet-stream",
        content_disposition_type="inline",
    )


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    document = kb.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    kb.delete_document(doc_id)
    Path(document.filepath).unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(doc_id: str, file: UploadFile = File(...)):
    if kb.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    tmp_path = _save_upload_to_temp(file)
    try:
        document = kb.update_document(doc_id, tmp_path, display_filename=file.filename)
        permanent_path = _persist_file(tmp_path, document.doc_id, file.filename)
        document.filepath = permanent_path
        kb.document_repository.save(document)
        return _to_document_response(document)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = supervisor.handle(request.question)

    sources: list[SourceResponse] = []
    for r in result.doc_sources:
        sources.append(
            SourceResponse(
                type="document",
                label=r.source_filename,
                page_number=r.chunk.page_number,
                score=r.score,
                text_snippet=r.chunk.text[:200],
            )
        )
    for w in result.web_sources:
        sources.append(
            SourceResponse(
                type="web",
                label=w.title,
                url=w.url,
                score=w.score,
                text_snippet=w.snippet[:200],
            )
        )

    return AskResponse(answer=result.text, answerable=result.answerable, sources=sources)


@app.post("/tts")
def text_to_speech(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        tts = gTTS(text=text, lang="en")
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc
    return StreamingResponse(buffer, media_type="audio/mpeg")