import pytest
from service.extractor.metadata_creator import MetadataCreator


class TestMetadataCreator:

    @pytest.fixture
    def metadata_creator(self):
        """Fixture for creating a MetadataCreator instance"""
        return MetadataCreator()

    @pytest.fixture
    def sample_extracted_result(self):
        """Fixture providing a sample extracted result dictionary"""
        return {
            "document_id": {"type": "string", "raw_value": "doc_001"},
            "policy_number": {
                "type": "string",
                "raw_value": "POL123456",
                "convert_value": "POL-123-456",  # Converted value priority
            },
            "policy_holder": {
                "name": {"type": "string", "raw_value": "John Doe"},
                "age": {"type": "number", "raw_value": "30", "convert_value": 30},
            },
            "insured": {"name": {"type": "string", "raw_value": "Jane Doe"}},
            "simple_field": "Simple Value",  # Case where it's not a dict with raw_value
        }

    def test_default_schema_loading(self, metadata_creator):
        """Test if the default schema is loaded correctly"""
        schema = metadata_creator.schema
        assert "document_id" in schema
        assert "policy_num" in schema
        assert "policy_holder_name" in schema
        assert "insured_name" in schema

        assert schema["policy_num"]["mapping"] == "policy_number"
        assert schema["policy_holder_name"]["mapping"] == ("policy_holder", "name")

    def test_create_metadata_basic(self, metadata_creator, sample_extracted_result):
        """Test basic metadata creation with default schema"""
        metadata = metadata_creator.create(sample_extracted_result)

        # Check mapped fields
        assert metadata["document_id"] == "doc_001"
        # policy_num maps to policy_number, which has convert_value
        assert metadata["policy_num"] == "POL-123-456"
        # Nested mapping
        assert metadata["policy_holder_name"] == "John Doe"
        assert metadata["insured_name"] == "Jane Doe"

    def test_create_metadata_missing_fields(self, metadata_creator):
        """Test metadata creation when some fields are missing in extracted result"""
        partial_result = {"document_id": {"raw_value": "doc_002"}}
        metadata = metadata_creator.create(partial_result)

        assert metadata["document_id"] == "doc_002"
        assert "policy_num" not in metadata
        assert "policy_holder_name" not in metadata

    def test_create_metadata_invalid_input(self, metadata_creator):
        """Test with invalid input type"""
        result = metadata_creator.create("not a dict")  # type: ignore
        assert result == {}

    def test_register_field_mapping(self, metadata_creator, sample_extracted_result):
        """Test registering a new field mapping and updating an existing one"""
        # Add new field
        metadata_creator.register_field_mapping("holder_age", ("policy_holder", "age"))

        # Update existing field mapping
        metadata_creator.register_field_mapping("document_id", "simple_field")

        metadata = metadata_creator.create(sample_extracted_result)

        assert metadata["holder_age"] == 30  # Should use convert_value
        assert metadata["document_id"] == "Simple Value"

    def test_add_schema_field(self, metadata_creator):
        """Test adding a full schema field configuration"""
        metadata_creator.add_schema_field("new_field", "int", mapping="custom_path")

        assert "new_field" in metadata_creator.schema
        assert metadata_creator.get_field_type("new_field") == "int"
        assert metadata_creator.get_field_mapping("new_field") == "custom_path"

        # Test default mapping
        metadata_creator.add_schema_field("another_field", "str")
        assert metadata_creator.get_field_mapping("another_field") == "another_field"

    def test_extract_value_priority(self, metadata_creator):
        """Test that convert_value is prioritized over raw_value"""
        data = {"test_field": {"raw_value": "raw", "convert_value": "converted"}}
        metadata_creator.register_field_mapping("test", "test_field")
        metadata = metadata_creator.create(data)
        assert metadata["test"] == "converted"

        data_raw_only = {"test_field": {"raw_value": "raw"}}
        metadata = metadata_creator.create(data_raw_only)
        assert metadata["test"] == "raw"

    def test_extract_direct_value(self, metadata_creator):
        """Test extracting a value that is not a dictionary structure"""
        data = {"simple": 123}
        metadata_creator.register_field_mapping("simple_val", "simple")
        metadata = metadata_creator.create(data)
        assert metadata["simple_val"] == 123

    def test_invalid_mappings(self, metadata_creator):
        """Test behavior with invalid mappings"""
        with pytest.raises(ValueError):
            metadata_creator.register_field_mapping("bad_field", 123)  # type: ignore
