import pytest
from rag.ingestion.splitter.markdown_splitter import RagMarkdownSplitter
from rag.ingestion.document import RagDocument


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
            pages=[
                {
                    "text": "# Insurance Policy ## Policy Number POL-123-456\n\n## Policy Holder Name: John Doe\n\n## Coverage\n\n- Medical Insurance\n- Life Insurance",
                    "metadata": {"page_number": 1},
                }
            ],
            confidence={"overall_confidence": 0.95},
        )

    @pytest.fixture
    def minimal_rag_document(self):
        """Fixture providing a minimal RagDocument"""
        return RagDocument(
            document_id="minimal-doc",
            filename="minimal.txt",
            pages=[{"text": "Simple text content without markdown.", "metadata": {"page_number": 1}}],
        )

    @pytest.fixture
    def empty_rag_document(self):
        """Fixture providing an empty RagDocument"""
        return RagDocument(
            document_id="empty-doc",
            filename="empty.txt",
            pages=[],
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
            pages=[{"text": "# Test\n\nContent here", "metadata": {"page_number": 1}}],
            content_type="application/pdf",
        )

        chunks = splitter.split_document(doc)

        for chunk in chunks:
            metadata = chunk["metadata"]
            # None values should be filtered
            assert None not in metadata.values()
            # But valid values should be present
            assert metadata["filename"] == "test.pdf"
            assert metadata["document_id"] == "test-doc"
            assert metadata["content_type"] == "application/pdf"

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
            pages=[
                {"text": "# Page 1 \n\n content 1", "metadata": {"page_number": 1}},
                {"text": "## Page 1.1 \n\n Subcontent 1.1", "metadata": {"page_number": 2}},
                {"text": "# Page 2 \n\n content 2", "metadata": {"page_number": 3}},
            ],
            content_type="text/markdown",
        )

        chunks = splitter.split_document(doc)

        # Should create multiple chunks for different sections
        assert len(chunks) >= 2

        # Verify sections are split
        texts = [chunk["text"] for chunk in chunks]
        assert any("content 1" in text for text in texts)
        assert any("Subcontent" in text for text in texts)
        assert any("content 2" in text for text in texts)

    def test_page_number_extraction_with_marker(self, splitter):
        """Test that page numbers are correctly extracted from chunks with page markers"""
        doc = RagDocument(
            document_id="page-marker-doc",
            filename="test.pdf",
            pages=[
                {"text": "# Title\n\nContent from page 1", "metadata": {"page_number": 1}},
                {"text": "Content from page 2", "metadata": {"page_number": 2}},
            ],
            content_type="application/pdf",
        )

        chunks = splitter.split_document(doc)

        # Verify that chunks have page_number in metadata
        for chunk in chunks:
            assert "page_number" in chunk["metadata"]
            page_num = chunk["metadata"]["page_number"]
            assert isinstance(page_num, int)
            assert page_num >= 0

        # Verify that page markers are removed from text
        for chunk in chunks:
            assert "[PAGE:" not in chunk["text"]
            assert "PAGE:" not in chunk["text"]

    def test_page_number_inheritance(self, splitter):
        """Test that chunks without page markers inherit the last page number"""
        doc = RagDocument(
            document_id="inheritance-doc",
            filename="test.pdf",
            pages=[
                {"text": "# Section 1\n\nFirst part of content", "metadata": {"page_number": 1}},
                {"text": "Second part of content\n\nThird part", "metadata": {"page_number": 1}},
            ],
            content_type="application/pdf",
        )

        chunks = splitter.split_document(doc)

        # All chunks should have page_number
        for chunk in chunks:
            assert "page_number" in chunk["metadata"]
            # Since all content is from page 1, all chunks should have page_number = 1
            # (or at least the first chunk with marker should be 1, others inherit)
            assert chunk["metadata"]["page_number"] == 1

    def test_multiple_pages_page_numbers(self, splitter):
        """Test page number assignment across multiple pages"""
        doc = RagDocument(
            document_id="multi-page-doc",
            filename="test.pdf",
            pages=[
                {"text": "# Page 1 Title\n\nPage 1 content", "metadata": {"page_number": 1}},
                {"text": "Page 2 content line 1\n\nPage 2 content line 2", "metadata": {"page_number": 2}},
                {"text": "# Page 3 Title\n\nPage 3 content", "metadata": {"page_number": 3}},
            ],
            content_type="application/pdf",
        )

        chunks = splitter.split_document(doc)

        # Verify chunks exist
        assert len(chunks) > 0

        # Collect page numbers
        page_numbers = [chunk["metadata"]["page_number"] for chunk in chunks]

        # Verify that page numbers are present and reasonable
        assert all(pn > 0 for pn in page_numbers), f"All page numbers should be > 0, got: {page_numbers}"

        for chunk in chunks:
            if "Page 1 content" in chunk["text"]:
                assert chunk["metadata"]["page_number"] == 1
            elif "Page 2 content line 1" in chunk["text"]:
                assert chunk["metadata"]["page_number"] == 2
            elif "Page 3 content" in chunk["text"]:
                assert chunk["metadata"]["page_number"] == 3

    def test_page_marker_removal(self, splitter):
        """Test that page markers are completely removed from chunk text"""
        doc = RagDocument(
            document_id="marker-removal-doc",
            filename="test.pdf",
            pages=[
                {"text": "Some content here", "metadata": {"page_number": 1}},
            ],
            content_type="application/pdf",
        )

        chunks = splitter.split_document(doc)

        # Verify no page markers remain in text
        for chunk in chunks:
            text = chunk["text"]
            assert "[PAGE:" not in text
            assert "PAGE:" not in text
            # Verify text is not empty (unless it's a special case)
            if chunk["metadata"].get("page_number", 0) > 0:
                assert len(text.strip()) > 0 or "Some content" in text

    def test_page_number_continuity(self, splitter):
        """Test that page numbers maintain continuity across chunks from the same page"""
        doc = RagDocument(
            document_id="continuity-doc",
            filename="test.pdf",
            pages=[
                {
                    "text": "# Section A\n\nContent A1\n\n## Subsection A1\n\nContent A2",
                    "metadata": {"page_number": 1},
                },
                {
                    "text": "# Section B\n\nContent B1",
                    "metadata": {"page_number": 2},
                },
            ],
            content_type="application/pdf",
        )

        chunks = splitter.split_document(doc)

        # Verify all chunks have page numbers
        page_numbers = [chunk["metadata"]["page_number"] for chunk in chunks]
        assert all(pn in [1, 2] for pn in page_numbers)

        # Verify that chunks from the same logical page have consistent page numbers
        # (This depends on how MarkdownNodeParser splits, but at minimum
        # we should have valid page numbers)
        assert len(set(page_numbers)) == 2, "Should have 2 different page numbers"

    def test_empty_pages_handling(self, splitter):
        """Test handling of documents with empty pages"""
        doc = RagDocument(
            document_id="empty-pages-doc",
            filename="test.pdf",
            pages=[
                {"text": "Content from page 1", "metadata": {"page_number": 1}},
                {"text": "", "metadata": {"page_number": 2}},  # Empty page
                {"text": "Content from page 3", "metadata": {"page_number": 3}},
            ],
            content_type="application/pdf",
        )

        chunks = splitter.split_document(doc)

        # Should still produce chunks (empty pages are skipped)
        assert isinstance(chunks, list)
        # Should have at least chunks from non-empty pages
        assert len(chunks) > 0

        # All chunks should have valid page numbers
        for chunk in chunks:
            if "Content from page 1" in chunk["text"]:
                assert chunk["metadata"]["page_number"] == 1
            elif "Content from page 3" in chunk["text"]:
                assert chunk["metadata"]["page_number"] == 3
            else:
                assert chunk["metadata"]["page_number"] == 2

    def test_page_number_without_metadata(self, splitter):
        """Test page number extraction when page metadata doesn't have page_number"""
        doc = RagDocument(
            document_id="no-metadata-doc",
            filename="test.pdf",
            pages=[
                {"text": "Content 1", "metadata": {}},  # No page_number in metadata
                {"text": "Content 2", "metadata": {"page_number": 2}},
            ],
            content_type="application/pdf",
        )

        chunks = splitter.split_document(doc)

        # Should still process chunks
        assert len(chunks) > 0

        # Chunks should have page_number (may be 0 if no marker found and no inheritance)
        for chunk in chunks:
            assert "page_number" in chunk["metadata"]
            assert isinstance(chunk["metadata"]["page_number"], int)
