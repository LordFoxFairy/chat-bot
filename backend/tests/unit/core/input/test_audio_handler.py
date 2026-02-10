import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from collections import deque
import time
import uuid

from backend.core.input.audio_handler import AudioInputHandler
from backend.core.models import AudioData, EventType, StreamEvent, TextData
from backend.core.session.session_context import SessionContext
from backend.core.interfaces.base_vad import BaseVAD
from backend.core.interfaces.base_asr import BaseASR

@pytest.fixture
def mock_session_context():
    context = MagicMock(spec=SessionContext)
    context.session_id = "test_session_audio"
    context.tag_id = "test_tag_audio"

    # Mock modules
    context.get_module = MagicMock()
    return context

@pytest.fixture
def mock_vad_module():
    vad = AsyncMock(spec=BaseVAD)
    vad.detect.return_value = False # Default to no speech
    return vad

@pytest.fixture
def mock_asr_module():
    asr = AsyncMock(spec=BaseASR)
    asr.recognize.return_value = "recognized text"
    return asr

@pytest.fixture
def mock_callback():
    return AsyncMock()

@pytest.fixture
def audio_handler(mock_session_context, mock_callback, mock_vad_module, mock_asr_module):
    # Setup context to return mocks
    def get_module_side_effect(module_name):
        if module_name == "vad":
            return mock_vad_module
        elif module_name == "asr":
            return mock_asr_module
        return None

    mock_session_context.get_module.side_effect = get_module_side_effect

    handler = AudioInputHandler(
        session_context=mock_session_context,
        result_callback=mock_callback,
        silence_timeout=0.1,        # Short timeout for testing
        max_buffer_duration=0.5,    # Short duration for testing
        min_segment_threshold=0.01  # Short threshold for testing
    )
    return handler

class TestAudioInputHandler:

    def test_initialization(self, audio_handler, mock_session_context, mock_vad_module, mock_asr_module):
        """Test initialization and module loading"""
        assert audio_handler.session_context == mock_session_context
        assert audio_handler.vad_module == mock_vad_module
        assert audio_handler.asr_module == mock_asr_module
        assert audio_handler.audio_buffer is not None
        assert audio_handler.is_processing is False

    @pytest.mark.asyncio
    async def test_start_stop(self, audio_handler):
        """Test start and stop lifecycle"""
        audio_handler.start()
        assert audio_handler.monitor_task is not None
        assert not audio_handler.monitor_task.done()

        await audio_handler.stop()
        assert audio_handler.monitor_task is None

    @pytest.mark.asyncio
    async def test_process_chunk_no_vad(self, audio_handler, mock_session_context):
        """Test behavior when VAD module is missing"""
        # Simulate missing VAD
        audio_handler.vad_module = None

        await audio_handler.process_chunk(b"audio_bytes")

        # Buffer should remain empty
        async with audio_handler.buffer_lock:
            assert len(audio_handler.audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_process_chunk_with_speech(self, audio_handler, mock_vad_module):
        """Test processing chunk when VAD detects speech"""
        mock_vad_module.detect.return_value = True
        chunk = b"speech_data"

        await audio_handler.process_chunk(chunk)

        async with audio_handler.buffer_lock:
            assert len(audio_handler.audio_buffer) == 1
            assert audio_handler.audio_buffer[0] == chunk
            assert audio_handler.last_speech_time is not None

    @pytest.mark.asyncio
    async def test_buffer_overflow_protection(self, audio_handler, mock_vad_module):
        """Test buffer overflow protection logic"""
        mock_vad_module.detect.return_value = True

        # First fill buffer
        small_chunk = b"12345"
        await audio_handler.process_chunk(small_chunk)

        # Override buffer and MAX_BUFFER_SIZE for test
        audio_handler.MAX_BUFFER_SIZE = 10
        # This chunk plus existing buffer (5) > 10, should clear buffer
        large_chunk = b"12345678"

        await audio_handler.process_chunk(large_chunk)

        async with audio_handler.buffer_lock:
            # Should only contain the new chunk after clearing
            assert len(audio_handler.audio_buffer) == 1
            assert audio_handler.audio_buffer[0] == large_chunk

    @pytest.mark.asyncio
    async def test_check_and_process_silence_timeout(self, audio_handler, mock_asr_module, mock_callback, mock_vad_module):
        """Test triggering processing via silence timeout"""
        mock_vad_module.detect.return_value = True

        # Set buffer above threshold (min_segment_threshold=0.01)
        # DEFAULT_BYTES_PER_SECOND = 32000
        # need > 320 bytes for 0.01s (buffer_duration check)
        # Mocking buffer directly to avoid VAD complexity for this test part
        chunk_size = 1000
        chunk = b"a" * chunk_size

        async with audio_handler.buffer_lock:
            audio_handler.audio_buffer.append(chunk)
            # Set last speech time to past (silence_timeout=0.1)
            audio_handler.last_speech_time = time.time() - 0.2

        # Trigger check
        await audio_handler._check_and_process(client_ended=False)

        # Verify ASR was called
        mock_asr_module.recognize.assert_called_once()

        # Verify callback (final result should be sent)
        mock_callback.assert_called_once()

        # Verify buffer cleared
        async with audio_handler.buffer_lock:
            assert len(audio_handler.audio_buffer) == 0
            assert audio_handler.last_speech_time is None

    @pytest.mark.asyncio
    async def test_signal_client_speech_end(self, audio_handler, mock_asr_module, mock_callback):
        """Test client signaling speech end"""
        # Add data to buffer
        async with audio_handler.buffer_lock:
            audio_handler.audio_buffer.append(b"data")

        audio_handler.signal_client_speech_end()
        assert audio_handler.client_speech_ended.is_set()

        # Manually trigger check since monitor loop isn't running in this isolation test
        await audio_handler._check_and_process(client_ended=True)

        # Should process as final
        mock_asr_module.recognize.assert_called_once()

        # Verify result sent
        assert mock_callback.call_count == 1
        event = mock_callback.call_args[0][0]
        assert event.event_data.is_final is True

    @pytest.mark.asyncio
    async def test_asr_processing_and_cleaning(self, audio_handler, mock_asr_module):
        """Test ASR processing and text cleaning"""
        mock_asr_module.recognize.return_value = " <|special|> Hello World "

        # Call private method directly for unit testing logic
        await audio_handler._process_audio_segment(b"audio", is_final=False)

        # Check transcript segments
        assert len(audio_handler.transcript_segments) == 1
        assert audio_handler.transcript_segments[0] == "Hello World"

    @pytest.mark.asyncio
    async def test_send_final_result(self, audio_handler, mock_callback):
        """Test sending final result"""
        audio_handler.transcript_segments = ["Hello", "World"]

        await audio_handler._send_final_result()

        assert mock_callback.call_count == 1
        event = mock_callback.call_args[0][0]

        assert event.event_data.text == "Hello World"
        assert event.event_data.is_final is True

        # Verify cleanup
        assert len(audio_handler.transcript_segments) == 0

    @pytest.mark.asyncio
    async def test_max_buffer_limit_trigger(self, audio_handler, mock_asr_module):
        """Test triggering processing via max buffer duration"""
        # max_buffer_duration=0.5
        # 0.5 * 32000 = 16000 bytes
        large_chunk = b"a" * 20000

        async with audio_handler.buffer_lock:
            audio_handler.audio_buffer.append(large_chunk)

        await audio_handler._check_and_process(client_ended=False)

        # Should process but NOT be final (since it's max buffer limit)
        mock_asr_module.recognize.assert_called_once()

        # Since it wasn't final, send_final_result shouldn't be called inside process_audio_segment
        # We can verify this via mock_callback call count if needed, but checking is_final logic is internal
        # Instead, verify transcript storage which happens on intermediate results
        assert len(audio_handler.transcript_segments) == 1

    def test_clean_text_pattern(self, audio_handler):
        """Test special token cleaning regex"""
        assert audio_handler._clean_text("<|start|>Hello<|end|>") == "Hello"
        assert audio_handler._clean_text("No special tokens") == "No special tokens"
        assert audio_handler._clean_text("") == ""
