import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from backend.core.input.text_handler import TextInputHandler
from backend.core.models import EventType, StreamEvent, TextData
from backend.core.session.session_context import SessionContext

@pytest.fixture
def mock_session_context():
    context = MagicMock(spec=SessionContext)
    context.session_id = "test_session_123"
    context.tag_id = "test_tag_456"
    return context

@pytest.fixture
def mock_callback():
    return AsyncMock()

@pytest.fixture
def text_handler(mock_session_context, mock_callback):
    return TextInputHandler(mock_session_context, mock_callback)

class TestTextInputHandler:

    def test_initialization(self, text_handler, mock_session_context):
        """Test initialization of TextInputHandler"""
        assert text_handler.session_context == mock_session_context
        assert text_handler.result_callback is not None

    @pytest.mark.asyncio
    async def test_process_text_valid(self, text_handler, mock_callback, mock_session_context):
        """Test processing valid text input"""
        input_text = "  Hello World  "
        expected_cleaned = "Hello World"

        await text_handler.process_text(input_text)

        # Verify callback was called
        assert mock_callback.call_count == 1

        # Verify event data
        call_args = mock_callback.call_args
        event, context = call_args[0]

        assert isinstance(event, StreamEvent)
        assert event.event_type == EventType.ASR_RESULT
        assert event.session_id == mock_session_context.session_id
        assert event.tag_id == mock_session_context.tag_id

        assert isinstance(event.event_data, TextData)
        assert event.event_data.text == expected_cleaned
        assert event.event_data.is_final is True

        # Verify context passed to callback matches what's expected
        assert context == {"session_id": mock_session_context.session_id}

    @pytest.mark.asyncio
    async def test_process_text_empty(self, text_handler, mock_callback):
        """Test processing empty text input (should be ignored)"""
        await text_handler.process_text("")
        await text_handler.process_text("   ")
        await text_handler.process_text(None)

        # Callback should not be called for empty/whitespace inputs
        assert mock_callback.call_count == 0

    def test_clean_text(self, text_handler):
        """Test text cleaning logic"""
        assert text_handler._clean_text("  hello  ") == "hello"
        assert text_handler._clean_text("\t\nhello\n") == "hello"
        assert text_handler._clean_text("") == ""
        assert text_handler._clean_text(None) == ""
