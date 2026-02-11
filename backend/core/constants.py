"""Core constants for the chat-bot application.

This module centralizes magic numbers and configuration defaults
to improve maintainability and reduce duplication.
"""

# =============================================================================
# Audio Constants
# =============================================================================

# Standard audio format for ASR
AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_CHANNELS = 1  # Mono
AUDIO_SAMPLE_WIDTH = 2  # 16-bit (2 bytes per sample)
AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH

# =============================================================================
# Buffer Constants
# =============================================================================

MAX_AUDIO_BUFFER_MB = 10
MAX_AUDIO_BUFFER_BYTES = MAX_AUDIO_BUFFER_MB * 1024 * 1024

# =============================================================================
# Noise Reduction Constants
# =============================================================================

HIGH_PASS_CUTOFF_HZ = 80  # Remove low frequency noise below this
LOW_PASS_CUTOFF_HZ = 7500  # Remove high frequency noise above this
NOISE_GATE_THRESHOLD_RATIO = 0.1  # 10% of RMS as noise gate threshold
NOISE_GATE_WINDOW_MS = 5  # Smoothing window for noise gate in milliseconds

# =============================================================================
# VAD (Voice Activity Detection) Constants
# =============================================================================

DEFAULT_VAD_THRESHOLD = 0.65
DEFAULT_MIN_SILENCE_DURATION_MS = 1500  # Silence duration to trigger end-of-speech
DEFAULT_MAX_SPEECH_SEGMENT_MS = 4000  # Maximum speech segment duration
DEFAULT_VAD_WINDOW_SIZE = 512  # VAD processing window size in samples

# =============================================================================
# Output Format Constants
# =============================================================================

OUTPUT_FORMAT_FLOAT32 = "pcm_f32le"  # Float32 little-endian
OUTPUT_FORMAT_INT16 = "pcm_s16le"  # Int16 little-endian

# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_CONFIG_PATH = "backend/configs/config.yaml"
DEFAULT_LOG_PATH = "logs/chat-bot.log"
DEFAULT_MODEL_OUTPUT_DIR = "outputs/models"

# =============================================================================
# WebSocket Constants
# =============================================================================

DEFAULT_WEBSOCKET_HOST = "0.0.0.0"
DEFAULT_WEBSOCKET_PORT = 8765
DEFAULT_WEBSOCKET_MAX_MESSAGE_SIZE = 2 * 1024 * 1024  # 2MB

# =============================================================================
# Session Constants
# =============================================================================

DEFAULT_ACTIVATION_TIMEOUT_SECONDS = 30
DEFAULT_RECONNECT_DELAY_MS = 3000
DEFAULT_MAX_RECONNECT_ATTEMPTS = 10
