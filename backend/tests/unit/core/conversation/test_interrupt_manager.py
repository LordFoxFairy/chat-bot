import pytest
from backend.core.conversation.interrupt_manager import InterruptManager

class TestInterruptManager:
    @pytest.fixture
    def manager(self):
        """Create an InterruptManager instance for testing."""
        return InterruptManager(session_id="test_session_123")

    def test_initial_state(self, manager):
        """Test initial state is clean."""
        assert manager.is_interrupted is False
        assert manager.was_interrupted is False
        assert manager.session_id == "test_session_123"

    def test_set_interrupt(self, manager):
        """Test setting interrupt state."""
        manager.set_interrupt()
        assert manager.is_interrupted is True
        assert manager.was_interrupted is True

    def test_reset(self, manager):
        """Test resetting current interrupt flag but keeping history."""
        manager.set_interrupt()

        # Verify state before reset
        assert manager.is_interrupted is True
        assert manager.was_interrupted is True

        manager.reset()

        # Verify state after reset
        assert manager.is_interrupted is False
        # Historical flag should remain True until explicitly reset
        assert manager.was_interrupted is True

    def test_reset_history(self, manager):
        """Test resetting history flag."""
        manager.set_interrupt()
        manager.reset()

        assert manager.was_interrupted is True

        manager.reset_history()
        assert manager.was_interrupted is False

    def test_interrupt_idempotency(self, manager):
        """Test that calling set_interrupt multiple times doesn't change behavior."""
        manager.set_interrupt()
        manager.set_interrupt()

        assert manager.is_interrupted is True
        assert manager.was_interrupted is True
