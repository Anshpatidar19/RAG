"""
Facade over the whole system. This is the single entry point the API
layer (or CLI) talks to — it doesn't know or care about Pinecone,
embeddings, or Gemini directly, just calls into this class.
"""

from core.models import Answer, Document
from generation.generator import AnswerGenerator
from ingestion.pipeline import IngestionPipeline
from retrieval.retriever import Retriever
from storage.document_repository import DocumentRepository
from storage.vector_store import VectorStoreRepository


class KnowledgeBase:
    def __init__(
        self,
        pipeline: IngestionPipeline,
        retriever: Retriever,
        generator: AnswerGenerator,
        vector_store: VectorStoreRepository,
        document_repository: DocumentRepository,
    ):
        self.pipeline = pipeline
        self.retriever = retriever
        self.generator = generator
        self.vector_store = vector_store
        self.document_repository = document_repository

    def add_document(self, filepath: str, display_filename: str | None = None) -> Document:
        document = self.pipeline.ingest(filepath, display_filename=display_filename)
        self.document_repository.save(document)
        return document

    def update_document(
        self, doc_id: str, filepath: str, display_filename: str | None = None
    ) -> Document:
        document = self.pipeline.update(doc_id, filepath, display_filename=display_filename)
        self.document_repository.save(document)
        return document

    def delete_document(self, doc_id: str) -> None:
        self.vector_store.delete_by_doc_id(doc_id)
        self.document_repository.delete(doc_id)

    def list_documents(self) -> list[Document]:
        return self.document_repository.list_all()

    def get_document(self, doc_id: str) -> Document | None:
        return self.document_repository.get(doc_id)

    def ask(self, query: str) -> Answer:
        results = self.retriever.retrieve(query)
        if not self.retriever.has_sufficient_context(results):
            from generation.generator import NOT_FOUND_MESSAGE

            return Answer(text=NOT_FOUND_MESSAGE, sources=[], answerable=False)
        return self.generator.generate(query, results)