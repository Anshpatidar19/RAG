"""
Persists conversations and messages in Supabase, so chat history
survives page refreshes and server restarts.
"""

from supabase import Client, create_client

from config import settings


class ConversationRepository:
    def __init__(self, client: Client | None = None):
        self._client = client or create_client(settings.supabase_url, settings.supabase_key)

    def create_conversation(self, conversation_id: str, title: str = "New chat") -> None:
        self._client.table("conversations").insert(
            {"id": conversation_id, "title": title}
        ).execute()

    def list_conversations(self) -> list[dict]:
        response = (
            self._client.table("conversations")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    def update_title(self, conversation_id: str, title: str) -> None:
        self._client.table("conversations").update({"title": title}).eq(
            "id", conversation_id
        ).execute()

    def delete_conversation(self, conversation_id: str) -> None:
        self._client.table("conversations").delete().eq("id", conversation_id).execute()

    def get_messages(self, conversation_id: str) -> list[dict]:
        response = (
            self._client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )
        return response.data

    def add_message(
        self, conversation_id: str, role: str, content: str, sources: list | None = None
    ) -> None:
        self._client.table("messages").insert(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "sources": sources or [],
            }
        ).execute()