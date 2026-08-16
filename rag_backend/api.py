"""
FastAPI entrypoint. Wraps the KnowledgeBase facade (for document CRUD)
and the Supervisor agent (for answering questions) in REST endpoints
for the React frontend to call. Run with:
    uvicorn api:app --reload
"""

import hashlib
import io
import json
import mimetypes
import re
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from gtts import gTTS
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel

from agents.greeting import GreetingAgent
from agents.reflection import ReflectionAgent
from agents.supervision import Supervisor
from config import settings
from embedding.embedder import Embedder
from generation.generator import AnswerGenerator
from ingestion.pipeline import IngestionPipeline
from ingestion.web_parser import WebFetchError, fetch_page
from kb.knowledge_base import KnowledgeBase
from retrieval.retriever import Retriever
from retrieval.vector_search import VectorSearchTool
from retrieval.web_search import WebSearchTool
from storage.conversation_repository import ConversationRepository
from storage.document_repository import DocumentRepository
from storage.vector_store import VectorStoreRepository

app = FastAPI(title="Universal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FILES_DIR = Path(__file__).parent / "uploaded_files"
FILES_DIR.mkdir(exist_ok=True)

# --- Wiring: document CRUD + chat history ---
_embedder = Embedder()
_vector_store = VectorStoreRepository()
_document_repository = DocumentRepository()
_conversation_repository = ConversationRepository()
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
    retriever=_retriever,
)


# --- Request/response schemas ---
class AskRequest(BaseModel):
    question: str
    history: list[dict] = []
    conversation_id: str | None = None


class TTSRequest(BaseModel):
    text: str


class UrlIngestRequest(BaseModel):
    url: str


class CreateConversationRequest(BaseModel):
    title: str = "New chat"


class SourceResponse(BaseModel):
    type: str
    label: str
    url: str | None = None
    page_number: int | None = None
    score: float | None = None
    text_snippet: str


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    sources: list[SourceResponse]
    conversation_id: str


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    num_chunks: int
    file_url: str
    duplicate: bool = False


# --- Helpers ---
def _to_document_response(doc, duplicate: bool = False) -> DocumentResponse:
    return DocumentResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        status=doc.status.value,
        num_chunks=doc.num_chunks,
        file_url=f"/documents/{doc.doc_id}/file",
        duplicate=duplicate,
    )


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _save_bytes_to_temp(content: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return tmp.name


def _persist_file(tmp_path: str, doc_id: str, original_filename: str) -> str:
    suffix = Path(original_filename).suffix
    dest = FILES_DIR / f"{doc_id}{suffix}"
    for existing in FILES_DIR.glob(f"{doc_id}.*"):
        existing.unlink(missing_ok=True)
    shutil.move(tmp_path, dest)
    return str(dest)


# --- Document endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...)):
    content = file.file.read()
    content_hash = _hash_bytes(content)

    existing = kb.document_repository.get_by_content_hash(content_hash)
    if existing is not None:
        # Identical file already indexed — skip parsing/chunking/embedding
        # entirely and just tell the frontend this is the existing document.
        return _to_document_response(existing, duplicate=True)

    tmp_path = _save_bytes_to_temp(content, Path(file.filename).suffix)
    try:
        document = kb.add_document(tmp_path, display_filename=file.filename)
        permanent_path = _persist_file(tmp_path, document.doc_id, file.filename)
        document.filepath = permanent_path
        document.content_hash = content_hash
        kb.document_repository.save(document)
        return _to_document_response(document)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents():
    return [_to_document_response(d) for d in kb.list_documents()]


@app.post("/documents/url", response_model=DocumentResponse)
def ingest_url(request: UrlIngestRequest):
    try:
        title, text = fetch_page(request.url)
    except WebFetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    content_hash = _hash_bytes(text.encode("utf-8"))
    existing = kb.document_repository.get_by_content_hash(content_hash)
    if existing is not None:
        return _to_document_response(existing, duplicate=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        document = kb.add_document(tmp_path, display_filename=title)
        document.filepath = request.url
        document.content_hash = content_hash
        kb.document_repository.save(document)
        return _to_document_response(document)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    document = kb.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.filepath.startswith("http://") or document.filepath.startswith("https://"):
        return RedirectResponse(url=document.filepath)

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
    if not (document.filepath.startswith("http://") or document.filepath.startswith("https://")):
        Path(document.filepath).unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(doc_id: str, file: UploadFile = File(...)):
    if kb.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    content = file.file.read()
    content_hash = _hash_bytes(content)
    tmp_path = _save_bytes_to_temp(content, Path(file.filename).suffix)
    try:
        document = kb.update_document(doc_id, tmp_path, display_filename=file.filename)
        permanent_path = _persist_file(tmp_path, document.doc_id, file.filename)
        document.filepath = permanent_path
        document.content_hash = content_hash
        kb.document_repository.save(document)
        return _to_document_response(document)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --- Conversation endpoints ---
@app.get("/conversations")
def list_conversations():
    return _conversation_repository.list_conversations()


@app.post("/conversations")
def create_conversation(request: CreateConversationRequest):
    conversation_id = str(uuid.uuid4())
    _conversation_repository.create_conversation(conversation_id, request.title)
    return {"id": conversation_id, "title": request.title}


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str):
    return _conversation_repository.get_messages(conversation_id)


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    _conversation_repository.delete_conversation(conversation_id)
    return {"deleted": conversation_id}


# --- Ask endpoints ---
def _ensure_conversation(conversation_id: str | None, first_message: str) -> str:
    if conversation_id:
        return conversation_id
    new_id = str(uuid.uuid4())
    title = first_message[:40] + ("…" if len(first_message) > 40 else "")
    _conversation_repository.create_conversation(new_id, title)
    return new_id


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    conversation_id = _ensure_conversation(request.conversation_id, request.question)
    try:
        result = supervisor.handle(request.question, history=request.history)
    except genai_errors.ServerError as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI model is temporarily overloaded. Please try again in a moment.",
        ) from exc
    except Exception as exc:
        # Any other unexpected failure (Pinecone connection drop, etc.) —
        # log the real traceback server-side for debugging, but return a
        # clean message to the client instead of an unhandled 500.
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail="Something went wrong reaching a backend service. Please try again.",
        ) from exc

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
                text_snippet=w.snippet[:200],
            )
        )

    trimmed_sources = sources[: settings.max_sources_shown]
    _conversation_repository.add_message(conversation_id, "user", request.question)
    _conversation_repository.add_message(
        conversation_id,
        "assistant",
        result.text,
        sources=[s.model_dump() for s in trimmed_sources],
    )

    return AskResponse(
        answer=result.text,
        answerable=result.answerable,
        sources=trimmed_sources,
        conversation_id=conversation_id,
    )


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    conversation_id = _ensure_conversation(request.conversation_id, request.question)

    def event_generator():
        yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"

        final_text = ""
        final_sources_payload = []

        try:
            for event in supervisor.handle_stream(request.question, history=request.history):
                if event["type"] == "token":
                    final_text += event["text"]
                if event["type"] == "sources":
                    sources_payload = []
                    for r in event["doc_sources"]:
                        sources_payload.append(
                            {
                                "type": "document",
                                "label": r.source_filename,
                                "url": None,
                                "page_number": r.chunk.page_number,
                                "score": r.score,
                                "text_snippet": r.chunk.text[:200],
                            }
                        )
                    for w in event["web_sources"]:
                        sources_payload.append(
                            {
                                "type": "web",
                                "label": w.title,
                                "url": w.url,
                                "page_number": None,
                                "text_snippet": w.snippet[:200],
                            }
                        )
                    final_sources_payload = sources_payload[: settings.max_sources_shown]
                    payload = {
                        "type": "sources",
                        "sources": final_sources_payload,
                        "answerable": event["answerable"],
                    }
                else:
                    payload = event
                yield f"data: {json.dumps(payload)}\n\n"
        except genai_errors.ServerError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'The AI model is temporarily overloaded. Please try again in a moment.'})}\n\n"
            return
        except Exception:
            # Any other unexpected failure (Pinecone connection drop, etc.) —
            # log the real traceback server-side, send a clean message to
            # the client instead of the stream just breaking silently.
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong reaching a backend service. Please try again.'})}\n\n"
            return

        _conversation_repository.add_message(conversation_id, "user", request.question)
        _conversation_repository.add_message(
            conversation_id, "assistant", final_text, sources=final_sources_payload
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/tts")
def text_to_speech(request: TTSRequest):
    text = _clean_for_speech(request.text)
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


def _clean_for_speech(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[_~]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()