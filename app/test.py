"""
Tests for app/sample.py

Run from the project root:
    pytest app/test.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

import sample
from sample import app, ingest_documents, generate_rag_response


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_collection(docs: list[str] | None = None) -> MagicMock:
    """Mock ChromaDB collection that returns *docs* when queried."""
    col = MagicMock()
    col.query.return_value = {"documents": [docs or []]}
    return col


def make_ollama_mock(*words: str) -> MagicMock:
    """
    Mock ollama.AsyncClient class whose instance's .chat() streams *words*
    as individual chunks.
    """

    async def _stream():
        for word in words:
            yield {"message": {"content": word}}

    mock_cls = MagicMock()
    mock_cls.return_value.chat = AsyncMock(return_value=_stream())
    return mock_cls


# ===========================================================================
# ingest_documents
# ===========================================================================


class TestIngestDocuments:
    def test_upserts_single_doc(self):
        col = make_collection()
        with patch("sample._collection", col):
            ingest_documents(["Some clinical text."], ["doc-1"])
        col.upsert.assert_called_once_with(
            documents=["Some clinical text."], ids=["doc-1"]
        )

    def test_upserts_multiple_docs_in_one_call(self):
        col = make_collection()
        with patch("sample._collection", col):
            ingest_documents(["a", "b", "c"], ["1", "2", "3"])
        col.upsert.assert_called_once_with(
            documents=["a", "b", "c"], ids=["1", "2", "3"]
        )

    def test_empty_lists_still_calls_upsert(self):
        col = make_collection()
        with patch("sample._collection", col):
            ingest_documents([], [])
        col.upsert.assert_called_once_with(documents=[], ids=[])


# ===========================================================================
# generate_rag_response  (unit tests on the async generator)
# ===========================================================================


class TestGenerateRagResponse:
    # ---- positive ----------------------------------------------------------

    async def test_yields_sse_formatted_chunks(self):
        col = make_collection(["Clinical trials are controlled studies."])
        ollama_cls = make_ollama_mock("Clinical", " trials", " matter.")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            chunks = [
                c async for c in generate_rag_response("What is a clinical trial?")
            ]

        assert len(chunks) > 0
        assert all(c.startswith("data: ") for c in chunks)

    async def test_joined_chunks_contain_streamed_words(self):
        col = make_collection(["context"])
        ollama_cls = make_ollama_mock("Hello", " world")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            chunks = [c async for c in generate_rag_response("query")]

        joined = "".join(c.removeprefix("data: ") for c in chunks)
        assert "Hello" in joined
        assert "world" in joined

    async def test_retrieved_docs_appear_in_system_prompt(self):
        col = make_collection(["Drug X reduces fever.", "Phase III complete."])
        ollama_cls = make_ollama_mock("answer")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            _ = [c async for c in generate_rag_response("Tell me about Drug X")]

        call_kwargs = ollama_cls.return_value.chat.call_args.kwargs
        system_msg = call_kwargs["messages"][0]["content"]
        assert "Drug X reduces fever." in system_msg
        assert "Phase III complete." in system_msg

    async def test_user_query_forwarded_to_llm(self):
        col = make_collection(["some context"])
        ollama_cls = make_ollama_mock("answer")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            _ = [c async for c in generate_rag_response("What is pharmacokinetics?")]

        call_kwargs = ollama_cls.return_value.chat.call_args.kwargs
        assert call_kwargs["messages"][1]["content"] == "What is pharmacokinetics?"

    async def test_empty_chunks_from_llm_are_skipped(self):
        col = make_collection(["context"])

        async def _stream_with_empty():
            yield {"message": {"content": "Hello"}}
            yield {"message": {"content": ""}}  # must be skipped
            yield {"message": {"content": " world"}}

        mock_cls = MagicMock()
        mock_cls.return_value.chat = AsyncMock(return_value=_stream_with_empty())

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", mock_cls),
        ):
            chunks = [c async for c in generate_rag_response("query")]

        assert len(chunks) == 2

    async def test_collection_queried_with_user_query(self):
        col = make_collection(["context"])
        ollama_cls = make_ollama_mock("answer")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            _ = [c async for c in generate_rag_response("drug safety")]

        col.query.assert_called_once_with(query_texts=["drug safety"], n_results=3)

    # ---- negative: empty context -------------------------------------------

    async def test_no_docs_uses_fallback_context_in_prompt(self):
        col = make_collection([])  # empty result set
        ollama_cls = make_ollama_mock("I do not have enough information.")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            _ = [c async for c in generate_rag_response("Unknown topic")]

        call_kwargs = ollama_cls.return_value.chat.call_args.kwargs
        system_msg = call_kwargs["messages"][0]["content"]
        assert "No relevant context found" in system_msg

    # ---- negative: error propagation ---------------------------------------

    async def test_chromadb_failure_yields_sse_error_event(self):
        col = MagicMock()
        col.query.side_effect = RuntimeError("ChromaDB unavailable")

        with patch("sample._collection", col):
            chunks = [c async for c in generate_rag_response("query")]

        assert len(chunks) == 1
        assert chunks[0].startswith("data: [ERROR]")
        assert "ChromaDB unavailable" in chunks[0]

    async def test_ollama_failure_yields_sse_error_event(self):
        col = make_collection(["some context"])
        mock_cls = MagicMock()
        mock_cls.return_value.chat = AsyncMock(
            side_effect=ConnectionError("Ollama not running")
        )

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", mock_cls),
        ):
            chunks = [c async for c in generate_rag_response("query")]

        assert len(chunks) == 1
        assert chunks[0].startswith("data: [ERROR]")
        assert "Ollama not running" in chunks[0]


# ===========================================================================
# /api/v1/chat/stream  (endpoint integration tests)
# ===========================================================================


class TestChatStreamEndpoint:
    # ---- positive ----------------------------------------------------------

    async def test_valid_request_returns_200(self):
        col = make_collection(["Clinical trial context."])
        ollama_cls = make_ollama_mock("A", " response.")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/chat/stream",
                    json={"query": "What is a clinical trial?", "session_id": "s-001"},
                )

        assert resp.status_code == 200

    async def test_response_has_event_stream_content_type(self):
        col = make_collection(["context"])
        ollama_cls = make_ollama_mock("response")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/chat/stream",
                    json={"query": "query", "session_id": "s-002"},
                )

        assert "text/event-stream" in resp.headers["content-type"]

    async def test_response_body_is_sse_format(self):
        col = make_collection(["context"])
        ollama_cls = make_ollama_mock("The", " answer", " is here.")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/chat/stream",
                    json={"query": "query", "session_id": "s-003"},
                )

        non_empty_lines = [ln for ln in resp.text.split("\n") if ln.strip()]
        assert all(ln.startswith("data: ") for ln in non_empty_lines)

    async def test_empty_string_query_is_accepted(self):
        """Empty string is valid per schema — LLM is still invoked."""
        col = make_collection([])
        ollama_cls = make_ollama_mock("I do not have enough information.")

        with (
            patch("sample._collection", col),
            patch("sample.ollama.AsyncClient", ollama_cls),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/chat/stream",
                    json={"query": "", "session_id": "s-004"},
                )

        assert resp.status_code == 200

    # ---- negative: HTTP method ---------------------------------------------

    async def test_get_method_not_allowed(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/chat/stream")

        assert resp.status_code == 405

    # ---- negative: validation errors (422) ---------------------------------

    async def test_missing_query_field_returns_422(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat/stream",
                json={"session_id": "s-005"},
            )
        assert resp.status_code == 422

    async def test_missing_session_id_returns_422(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat/stream",
                json={"query": "What is a clinical trial?"},
            )
        assert resp.status_code == 422

    async def test_empty_payload_returns_422(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/v1/chat/stream", json={})
        assert resp.status_code == 422

    async def test_non_json_body_returns_422(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat/stream",
                content="not json",
                headers={"Content-Type": "text/plain"},
            )
        assert resp.status_code == 422

    async def test_null_query_returns_422(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat/stream",
                json={"query": None, "session_id": "s-006"},
            )
        assert resp.status_code == 422

    async def test_null_session_id_returns_422(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat/stream",
                json={"query": "clinical trial", "session_id": None},
            )
        assert resp.status_code == 422
