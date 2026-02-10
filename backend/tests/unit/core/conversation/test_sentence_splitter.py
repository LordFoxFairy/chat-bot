import pytest
from backend.core.conversation.sentence_splitter import SentenceSplitter

class TestSentenceSplitter:
    @pytest.fixture
    def splitter(self):
        """Create a SentenceSplitter instance for testing."""
        return SentenceSplitter()

    def test_initial_state(self, splitter):
        """Test initial state is empty."""
        assert splitter.buffer == ""
        assert splitter.get_remaining() == ""

    def test_simple_split(self, splitter):
        """Test simple sentence splitting."""
        splitter.append("Hello world.")
        sentence = splitter.split()
        assert sentence == "Hello world."
        assert splitter.buffer == ""

    def test_incomplete_sentence(self, splitter):
        """Test appended text without punctuation."""
        splitter.append("Hello world")
        sentence = splitter.split()
        assert sentence is None
        assert splitter.buffer == "Hello world"

    def test_multiple_sentences_in_one_append(self, splitter):
        """Test multiple sentences in a single append."""
        splitter.append("First sentence. Second sentence!")

        s1 = splitter.split()
        assert s1 == "First sentence."

        s2 = splitter.split()
        assert s2 == " Second sentence!"

        assert splitter.split() is None

    def test_incremental_append(self, splitter):
        """Test building a sentence across multiple appends."""
        splitter.append("This is")
        assert splitter.split() is None

        splitter.append(" a test")
        assert splitter.split() is None

        splitter.append(".")
        assert splitter.split() == "This is a test."

    def test_chinese_punctuation(self, splitter):
        """Test splitting with Chinese punctuation."""
        splitter.append("你好，世界！")

        s1 = splitter.split()
        assert s1 == "你好，"

        s2 = splitter.split()
        assert s2 == "世界！"

        assert splitter.split() is None

    def test_mixed_punctuation(self, splitter):
        """Test various supported punctuation marks."""
        text = "One, two. Three? Four! Five; Six:"
        # Note: colon is not in default pattern r'([，。！？；、,.!?;])'

        splitter.append(text)

        assert splitter.split() == "One,"
        assert splitter.split() == " two."
        assert splitter.split() == " Three?"
        assert splitter.split() == " Four!"
        assert splitter.split() == " Five;"

        # Remaining part
        assert splitter.split() is None
        assert splitter.buffer == " Six:"

    def test_get_remaining(self, splitter):
        """Test retrieving remaining buffer content."""
        splitter.append("Incomplete sentence")
        remaining = splitter.get_remaining()
        assert remaining == "Incomplete sentence"
        assert splitter.buffer == ""

    def test_clear(self, splitter):
        """Test clearing the buffer."""
        splitter.append("Some text")
        splitter.clear()
        assert splitter.buffer == ""
        assert splitter.split() is None

    def test_custom_pattern(self):
        """Test using a custom delimiter pattern."""
        # Using pipe as delimiter
        splitter = SentenceSplitter(delimiter_pattern=r'(\|)')

        splitter.append("Part 1|Part 2")
        assert splitter.split() == "Part 1|"
        assert splitter.buffer == "Part 2"

    def test_empty_append(self, splitter):
        """Test appending empty string (should be ignored)."""
        splitter.append("")
        assert splitter.buffer == ""

        splitter.append(None)
        assert splitter.buffer == ""

    def test_buffer_retention_after_split(self, splitter):
        """Test that buffer retains unsplit content correctly."""
        splitter.append("Sentence one. Beginning of two")
        assert splitter.split() == "Sentence one."
        assert splitter.buffer == " Beginning of two"

        splitter.append(" is complete.")
        assert splitter.split() == " Beginning of two is complete."
        assert splitter.buffer == ""
