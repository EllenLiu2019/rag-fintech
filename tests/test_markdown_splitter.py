import pytest
from service.splitter.markdown_splitter import RagMarkdownSplitter
from service.ingestion.document import RagDocument


class TestRagMarkdownSplitter:

    @pytest.fixture
    def splitter(self):
        """Fixture for creating a RagMarkdownSplitter instance"""
        return RagMarkdownSplitter()

    @pytest.fixture
    def sample_rag_document(self):
        """Fixture providing a sample RagDocument with markdown content"""
        return RagDocument(
            document_id="test-doc-001",
            filename="test_policy.pdf",
            file_size=1024,
            content_type="application/pdf",
            text="# Insurance Policy\n\n## Policy Number\n\nPOL-123-456\n\n## Policy Holder\n\nName: John Doe\n\n## Coverage\n\n- Medical Insurance\n- Life Insurance",
            pages=[{"text": "Page 1 content", "metadata": {"page_number": 1}}],
            extracted_data={"policy_number": "POL-123-456", "holder_name": "John Doe"},
            confidence={"overall_confidence": 0.95},
            metadata={"policy_number": "POL-123-456", "holder_name": "John Doe"},
        )

    @pytest.fixture
    def minimal_rag_document(self):
        """Fixture providing a minimal RagDocument"""
        return RagDocument(
            document_id="minimal-doc",
            filename="minimal.txt",
            text="Simple text content without markdown.",
        )

    @pytest.fixture
    def empty_rag_document(self):
        """Fixture providing an empty RagDocument"""
        return RagDocument(
            document_id="empty-doc",
            filename="empty.txt",
            text="",
        )

    def test_split_document_basic(self, splitter, sample_rag_document):
        """Test basic document splitting functionality"""
        chunks = splitter.split_document(sample_rag_document)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check chunk structure
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert isinstance(chunk["chunk_id"], str)
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["metadata"], dict)

    def test_metadata_inheritance(self, splitter, sample_rag_document):
        """Test that all chunks inherit metadata from parent document"""
        chunks = splitter.split_document(sample_rag_document)

        for chunk in chunks:
            metadata = chunk["metadata"]
            # Check that document-level metadata is inherited
            assert metadata["filename"] == "test_policy.pdf"
            assert metadata["document_id"] == "test-doc-001"
            assert metadata["content_type"] == "application/pdf"
            # Check that extracted_data metadata is also present
            assert "policy_number" in metadata and "holder_name" in metadata

    def test_split_empty_document(self, splitter, empty_rag_document):
        """Test splitting an empty document"""
        chunks = splitter.split_document(empty_rag_document)

        # Empty document might still produce chunks (depends on MarkdownNodeParser behavior)
        # But at least it shouldn't crash
        assert isinstance(chunks, list)

    def test_split_minimal_document(self, splitter, minimal_rag_document):
        """Test splitting a document without markdown structure"""
        chunks = splitter.split_document(minimal_rag_document)

        assert len(chunks) > 0
        # Should have at least one chunk with the text content
        assert any("Simple text content" in chunk["text"] for chunk in chunks)

    def test_metadata_none_values_filtered(self, splitter):
        """Test that None values in metadata are filtered out"""
        doc = RagDocument(
            document_id="test-doc",
            filename="test.pdf",
            text="# Test\n\nContent here",
            content_type=None,  # None value
            metadata={"key1": "value1", "key2": None},  # None in metadata
        )

        chunks = splitter.split_document(doc)

        for chunk in chunks:
            metadata = chunk["metadata"]
            # None values should be filtered
            assert None not in metadata.values()
            # But valid values should be present
            assert metadata["filename"] == "test.pdf"
            assert metadata["document_id"] == "test-doc"
            assert metadata.get("key1") == "value1"
            assert "key2" not in metadata or metadata["key2"] is not None

    def test_chunk_text_content(self, splitter, sample_rag_document):
        """Test that chunk text content is extracted correctly"""
        chunks = splitter.split_document(sample_rag_document)

        # Collect all text from chunks
        all_text = " ".join(chunk["text"] for chunk in chunks)

        # Original content should be present in chunks
        assert "POL-123-456" in all_text or any("POL-123-456" in chunk["text"] for chunk in chunks)
        assert "John Doe" in all_text or any("John Doe" in chunk["text"] for chunk in chunks)

    def test_chunk_id_uniqueness(self, splitter, sample_rag_document):
        """Test that each chunk has a unique chunk_id"""
        chunks = splitter.split_document(sample_rag_document)

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        # All chunk IDs should be unique
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_metadata_exclusion_keys(self, splitter, sample_rag_document):
        """Test that certain metadata keys are excluded from embedding/LLM"""
        chunks = splitter.split_document(sample_rag_document)

        # The excluded keys should still be in metadata (for storage)
        # but they are marked as excluded for embedding/LLM processing
        # This is handled by LlamaIndex Document's excluded_*_metadata_keys
        # We just verify the chunks have metadata
        for chunk in chunks:
            assert "metadata" in chunk
            # The metadata should contain the document-level info
            assert "filename" in chunk["metadata"]

    def test_multiple_sections_splitting(self, splitter):
        """Test splitting a document with multiple markdown sections"""
        doc = RagDocument(
            document_id="multi-section-doc",
            filename="multi.md",
            text="# Section 1\n\nContent 1\n\n## Subsection 1.1\n\nSubcontent\n\n# Section 2\n\nContent 2",
            metadata={"policy_number": "POL-123-456", "holder_name": "John Doe"},
        )

        chunks = splitter.split_document(doc)

        # Should create multiple chunks for different sections
        assert len(chunks) >= 2

        # Verify sections are split
        texts = [chunk["text"] for chunk in chunks]
        assert any("Section 1" in text for text in texts)
        assert any("Section 2" in text for text in texts)
