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
            "effective_date": {"type": "string", "raw_value": "2025-01-01"},
            "expiry_date": {"type": "string", "raw_value": "2026-01-01"},
            "coverage": [
                {
                    "cvg_name": {
                        "type": "string",
                        "raw_value": "个人癌症医疗保险（互联网2022版A款）",
                        "match": {"strategy": "exact", "values": ["保险名称"]},
                        "transform": None,
                    },
                    "cvg_type": {
                        "type": "string",
                        "raw_value": "恶性肿瘤质子重离子医疗保险金",
                        "match": {"strategy": "exact", "values": ["保险责任"]},
                        "transform": None,
                    },
                    "cvg_amt": {
                        "type": "number",
                        "raw_value": "2,000,000'",
                        "match": {"strategy": "exact", "values": ["最高保险金额（元）"]},
                        "transform": {"type": ["remove_commas", "remove_currency"]},
                        "convert_value": 2000000.0,
                    },
                    "description": {
                        "type": "string",
                        "raw_value": "次投保或非连续投保等待期:90天免赔额:0元/年社保目录内医疗费用赔付比例:100%社保目录外医疗费用赔付比例:100%",
                        "match": {"strategy": "exact", "values": ["详细说明"]},
                        "transform": None,
                    },
                }
            ],
            "cvg_premium": [
                {
                    "cvg_name": {
                        "type": "string",
                        "raw_value": "个人癌症医疗保险（互联网2022版A款）",
                        "match": {"strategy": "exact", "values": ["条款名称"]},
                        "transform": None,
                    },
                    "premium": {
                        "type": "number",
                        "raw_value": "2,284.00",
                        "match": {"strategy": "exact", "values": ["保险费（元）"]},
                        "transform": {"type": ["remove_commas"]},
                        "convert_value": 2284.0,
                    },
                }
            ],
        }

    def test_default_schema_loading(self, metadata_creator):
        """Test if the default schema is loaded correctly"""
        schema = metadata_creator.schema
        assert "document_id" in schema
        assert "policy_number" in schema
        assert "holder_name" in schema
        assert "insured_name" in schema

        assert schema["policy_number"]["mapping"] == "policy_number"
        assert schema["holder_name"]["mapping"] == ("policy_holder", "name")

    def test_create_metadata_basic(self, metadata_creator, sample_extracted_result):
        """Test basic metadata creation with default schema"""
        metadata = metadata_creator.create(sample_extracted_result)

        # Check mapped fields
        assert metadata["document_id"] == "doc_001"
        # policy_num maps to policy_number, which has convert_value
        assert metadata["policy_number"] == "POL-123-456"
        # Nested mapping
        assert metadata["holder_name"] == "John Doe"
        assert metadata["insured_name"] == "Jane Doe"

    def test_create_metadata_missing_fields(self, metadata_creator):
        """Test metadata creation when some fields are missing in extracted result"""
        partial_result = {"document_id": {"raw_value": "doc_002"}}
        metadata = metadata_creator.create(partial_result)

        assert metadata["document_id"] == "doc_002"
        assert "policy_number" not in metadata
        assert "holder_name" not in metadata

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

        # Prepare data that has simple_field
        sample_extracted_result["simple_field"] = {"raw_value": "Simple Value"}

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

    def test_create_metadata_with_embed_text(self, metadata_creator):
        """Test embed_metadata generation from embed_text"""
        data = {
            "policy_number": {"type": "string", "raw_value": "123456", "embed_text": "Policy No"},
            "policy_holder": {"name": {"type": "string", "raw_value": "John", "embed_text": "Holder Name"}},
            "coverage": [{"name": {"raw_value": "Health", "embed_text": "Cvg Name"}}],
        }

        metadata = metadata_creator.create(data)
        assert "embed_metadata" in metadata
        embed_str = metadata["embed_metadata"]
        assert "Policy No:123456" in embed_str
        assert "Holder Name:John" in embed_str
        assert "Cvg Name:Health" in embed_str
