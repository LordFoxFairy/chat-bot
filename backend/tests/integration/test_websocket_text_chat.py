"""
WebSocket Text Chat Integration Test

This integration test verifies the end-to-end flow of text-based conversation:
1. Connect to WebSocket server
2. Register a client session
3. Send a text message (CLIENT_TEXT_INPUT)
4. Receive and validate:
   - SERVER_TEXT_RESPONSE (streamed text replies from LLM)
   - SERVER_AUDIO_RESPONSE (TTS audio data)
5. Verify the response is meaningful and audio data is non-empty

Requirements:
- WebSocket server must be running on the specified host:port
- Real API key must be configured for LLM
- No mocking - uses real code and real APIs

Usage:
    # Start the server first:
    export API_KEY=your-api-key
    python3 backend/main.py server

    # Then run the test:
    python3 -m pytest backend/tests/integration/test_websocket_text_chat.py -v

    # Or run directly:
    python3 backend/tests/integration/test_websocket_text_chat.py --host localhost --port 8765
"""

import asyncio
import time
import uuid
import sys
import os
import argparse
from typing import List, Optional

import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import websockets
from backend.core.models.stream_event import StreamEvent, EventType
from backend.core.models.text_data import TextData
from backend.core.models.audio_data import AudioData


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def print_step(msg: str) -> None:
    """Print a step header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 20} {msg} {'=' * 20}{Colors.RESET}\n")


def print_result(msg: str, success: bool = True) -> None:
    """Print a test result"""
    color = Colors.GREEN if success else Colors.RED
    mark = "[PASS]" if success else "[FAIL]"
    print(f"{color}{mark} {msg}{Colors.RESET}")


def print_info(msg: str) -> None:
    """Print info message"""
    print(f"{Colors.CYAN}[INFO] {msg}{Colors.RESET}")


class WebSocketChatTestResult:
    """Test result container"""

    def __init__(self):
        self.session_id: Optional[str] = None
        self.text_responses: List[str] = []
        self.audio_chunks_count: int = 0
        self.audio_total_bytes: int = 0
        self.received_final_text: bool = False
        self.received_audio: bool = False
        self.errors: List[str] = []

    @property
    def full_response(self) -> str:
        return "".join(self.text_responses)

    @property
    def is_success(self) -> bool:
        return (
            len(self.full_response) > 0
            and self.audio_chunks_count > 0
            and self.audio_total_bytes > 0
            and len(self.errors) == 0
        )


async def run_websocket_chat_test(
    host: str = "localhost",
    port: int = 8765,
    test_message: str = "你好",
    timeout: float = 60.0,
    verbose: bool = True,
) -> WebSocketChatTestResult:
    """
    Run the WebSocket chat integration test.

    Args:
        host: WebSocket server host
        port: WebSocket server port
        test_message: Message to send for testing
        timeout: Total timeout in seconds
        verbose: Whether to print detailed output

    Returns:
        WebSocketChatTestResult containing test results
    """
    result = WebSocketChatTestResult()
    uri = f"ws://{host}:{port}"

    if verbose:
        print_info(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri, max_size=10 * 1024 * 1024) as websocket:
            if verbose:
                print_result("Connected to WebSocket Server")

            # Step 1: Register session
            if verbose:
                print_step("Sending SYSTEM_CLIENT_SESSION_START")

            tag_id = f"test_client_{uuid.uuid4().hex[:8]}"
            register_event = StreamEvent(
                event_type=EventType.SYSTEM_CLIENT_SESSION_START,
                tag_id=tag_id,
                timestamp=time.time(),
            )

            await websocket.send(register_event.to_json())
            if verbose:
                print_info(f"Sent registration: tag_id={tag_id}")

            # Wait for registration confirmation
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            session_start_event = StreamEvent.from_json(response)

            if session_start_event.event_type != EventType.SYSTEM_SERVER_SESSION_START:
                result.errors.append(
                    f"Expected SYSTEM_SERVER_SESSION_START, got {session_start_event.event_type}"
                )
                return result

            result.session_id = session_start_event.session_id
            if verbose:
                print_result(f"Registration successful. Session ID: {result.session_id}")

            # Step 2: Send text message
            if verbose:
                print_step("Sending CLIENT_TEXT_INPUT")

            input_event = StreamEvent(
                event_type=EventType.CLIENT_TEXT_INPUT,
                event_data=TextData(text=test_message, is_final=True),
                session_id=result.session_id,
                tag_id=tag_id,
                timestamp=time.time(),
            )

            await websocket.send(input_event.to_json())
            if verbose:
                print_info(f"Sent message: '{test_message}'")

            # Step 3: Receive responses
            if verbose:
                print_step("Waiting for Responses")

            start_time = time.time()
            message_timeout = 10.0  # Timeout for individual messages
            no_message_count = 0
            max_no_message = 3  # Allow 3 consecutive timeouts after receiving some data

            while time.time() - start_time < timeout:
                try:
                    raw_msg = await asyncio.wait_for(
                        websocket.recv(), timeout=message_timeout
                    )

                    if not raw_msg:
                        continue

                    if isinstance(raw_msg, str):
                        event = StreamEvent.from_json(raw_msg)

                        if event.event_type == EventType.SERVER_TEXT_RESPONSE:
                            if isinstance(event.event_data, TextData):
                                text_chunk = event.event_data.text
                                result.text_responses.append(text_chunk)
                                if verbose:
                                    is_final = event.event_data.is_final
                                    print(f"[TEXT] {text_chunk} (Final: {is_final})")
                                if event.event_data.is_final:
                                    result.received_final_text = True
                                    if verbose:
                                        print_result("Received complete text response")

                        elif event.event_type == EventType.SERVER_AUDIO_RESPONSE:
                            if isinstance(event.event_data, AudioData):
                                audio_chunk_size = len(event.event_data.data)
                                result.audio_chunks_count += 1
                                result.audio_total_bytes += audio_chunk_size
                                result.received_audio = True
                                if verbose:
                                    sys.stdout.write(
                                        f"\r[AUDIO] Chunk #{result.audio_chunks_count}: "
                                        f"{audio_chunk_size} bytes (Total: {result.audio_total_bytes} bytes)"
                                    )
                                    sys.stdout.flush()

                        elif event.event_type == EventType.STREAM_END:
                            if verbose:
                                print(f"\n[STREAM_END] Received")
                            if result.received_final_text and result.received_audio:
                                break

                        elif event.event_type == EventType.ERROR:
                            error_text = (
                                event.event_data.text
                                if isinstance(event.event_data, TextData)
                                else str(event.event_data)
                            )
                            result.errors.append(f"Server error: {error_text}")
                            if verbose:
                                print_result(f"Server Error: {error_text}", success=False)

                    # Reset no-message counter on successful receive
                    no_message_count = 0

                except asyncio.TimeoutError:
                    no_message_count += 1
                    if verbose:
                        print(f"\n[TIMEOUT] No message for {message_timeout} seconds...")

                    # If we've received both text and audio, we can exit
                    if result.received_final_text or (
                        result.received_audio and no_message_count >= max_no_message
                    ):
                        if verbose:
                            print("[INFO] Response appears complete.")
                        break

            if verbose:
                print("\n")

    except ConnectionRefusedError:
        result.errors.append(f"Connection refused to {uri}. Is the server running?")
        if verbose:
            print_result(
                f"Connection Refused to {uri}. Is the server running?", success=False
            )
    except Exception as e:
        result.errors.append(f"{type(e).__name__}: {e}")
        if verbose:
            print_result(f"Exception: {type(e).__name__}: {e}", success=False)

    return result


async def test_websocket_text_chat_async(
    host: str = "localhost", port: int = 8765
) -> bool:
    """
    Async test function for WebSocket text chat.

    Returns:
        True if test passes, False otherwise
    """
    print_step("Starting WebSocket Text Chat Integration Test")

    result = await run_websocket_chat_test(
        host=host,
        port=port,
        test_message="你好",
        timeout=60.0,
        verbose=True,
    )

    # Verification
    print_step("Verification Results")

    print(f"Full Response: {Colors.BOLD}{result.full_response}{Colors.RESET}")

    # Check 1: Text response is meaningful
    if result.full_response and len(result.full_response) > 0:
        print_result(f"Text Response Valid (Length: {len(result.full_response)})")
    else:
        print_result("Text Response Empty or Missing", success=False)

    # Check 2: Audio data is non-empty
    if result.audio_chunks_count > 0 and result.audio_total_bytes > 0:
        print_result(
            f"Audio Response Valid ({result.audio_chunks_count} chunks, "
            f"{result.audio_total_bytes} bytes)"
        )
    else:
        print_result("Audio Response Missing", success=False)

    # Check 3: No errors
    if result.errors:
        for error in result.errors:
            print_result(f"Error: {error}", success=False)

    return result.is_success


# Pytest test function
@pytest.mark.asyncio
@pytest.mark.integration
async def test_websocket_text_chat():
    """
    pytest integration test for WebSocket text chat.

    This test requires:
    - WebSocket server running on localhost:8765
    - Valid API key configured

    Skip this test if the server is not running.
    """
    host = os.environ.get("TEST_WS_HOST", "localhost")
    port = int(os.environ.get("TEST_WS_PORT", "8765"))

    # Try to connect first to check if server is available
    try:
        async with websockets.connect(
            f"ws://{host}:{port}", close_timeout=2
        ) as ws:
            pass
    except Exception:
        pytest.skip(f"WebSocket server not available at {host}:{port}")

    result = await run_websocket_chat_test(
        host=host,
        port=port,
        test_message="你好",
        timeout=60.0,
        verbose=False,
    )

    assert result.session_id is not None, "Session registration failed"
    assert len(result.full_response) > 0, "No text response received"
    assert result.audio_chunks_count > 0, "No audio chunks received"
    assert result.audio_total_bytes > 0, "No audio data received"
    assert len(result.errors) == 0, f"Errors occurred: {result.errors}"


def main():
    """Main entry point for direct execution"""
    parser = argparse.ArgumentParser(
        description="WebSocket Text Chat Integration Test"
    )
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    args = parser.parse_args()

    try:
        success = asyncio.run(
            test_websocket_text_chat_async(host=args.host, port=args.port)
        )
        if success:
            print_step("TEST PASSED")
            sys.exit(0)
        else:
            print_step("TEST FAILED")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest stopped by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
