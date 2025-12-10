import pytest
from rag.ingestion.extractor.metadata_creator import MetadataCreator


class TestMetadataCreator:

    @pytest.fixture
    def metadata_creator(self):
        """Fixture for creating a MetadataCreator instance"""
        return MetadataCreator()

    @pytest.fixture
    def sample_extracted_result(self):
        """
        Fixture providing a sample extracted result dictionary.
        Format matches RuleExtractor output:
        - Simple fields: {"中文字段名": "值"}
        - Nested fields: {"中文字段名": "值", ...}
        - List fields: [{"字段1": "值1", "字段2": "值2"}, ...]
        """
        return {
            "policy_number": {"保单号": "POL123456"},
            "policy_holder": {
                "投保人": "John Doe",
                "性别": "男",
                "出生日期": "1990-01-01",
                "证件号码": "123456789012345678",
            },
            "insured": {
                "被保险人": "Jane Doe",
                "性别": "女",
                "出生日期": "1985-05-15",
                "证件号码": "987654321098765432",
                "与投保人关系": "配偶",
            },
            "effective_date": {"保险期间开始日期": "2025-01-01"},
            "expiry_date": {"保险期间结束日期": "2026-01-01"},
            "coverage": [
                {
                    "保险名称": "个人癌症医疗保险（互联网2022版A款）",
                    "保险责任": "恶性肿瘤质子重离子医疗保险金",
                    "最高保险金额（元）": "2,000,000",
                    "详细说明": "首次投保或非连续投保等待期:90天",
                }
            ],
            "cvg_premium": [
                {
                    "条款名称": "个人癌症医疗保险（互联网2022版A款）",
                    "保险费（元）": "2,284.00",
                }
            ],
        }

    def test_default_schema_loading(self, metadata_creator):
        """Test if the default schema is loaded correctly"""
        schema = metadata_creator.schema
        # Check key fields exist in schema
        assert "policy_number" in schema
        assert "holder_name" in schema
        assert "insured_name" in schema
        assert "effective_date" in schema
        assert "coverage" in schema

        # Check mapping configuration
        assert schema["policy_number"]["mapping"] == "policy_number"
        assert schema["holder_name"]["mapping"] == ("policy_holder", "投保人")

    def test_create_metadata_basic(self, metadata_creator, sample_extracted_result):
        """Test basic metadata creation with default schema"""
        metadata = metadata_creator.create(sample_extracted_result)

        # policy_number: direct mapping, takes first value from dict
        assert metadata["policy_number"] == "POL123456"
        # holder_name: nested mapping (policy_holder, 投保人)
        assert metadata["holder_name"] == "John Doe"
        # insured_name: nested mapping (insured, 被保险人)
        assert metadata["insured_name"] == "Jane Doe"
        # effective_date: direct mapping, takes first value
        assert metadata["effective_date"] == "2025-01-01"
        # coverage: list type, returns the list directly
        assert isinstance(metadata["coverage"], list)

    def test_create_metadata_missing_fields(self, metadata_creator):
        """Test metadata creation when some fields are missing in extracted result"""
        partial_result = {"policy_number": {"保单号": "doc_002"}}
        metadata = metadata_creator.create(partial_result)

        assert metadata["policy_number"] == "doc_002"
        assert "holder_name" not in metadata
        assert "insured_name" not in metadata

    def test_create_metadata_invalid_input(self, metadata_creator):
        """Test with invalid input type"""
        result = metadata_creator.create("not a dict")  # type: ignore
        assert result == {}

    def test_register_field_mapping(self, metadata_creator, sample_extracted_result):
        """Test registering a new field mapping and updating an existing one"""
        # Add new field mapping with nested path
        metadata_creator.register_field_mapping("holder_gender", ("policy_holder", "性别"))

        # Add simple field mapping
        metadata_creator.register_field_mapping("custom_field", "simple_field")
        sample_extracted_result["simple_field"] = {"键": "Simple Value"}

        metadata = metadata_creator.create(sample_extracted_result)

        # Test nested mapping
        assert metadata["holder_gender"] == "男"
        # Test simple mapping (takes first value from dict)
        assert metadata["custom_field"] == "Simple Value"

    def test_add_schema_field(self, metadata_creator):
        """Test adding a full schema field configuration"""
        metadata_creator.add_schema_field("new_field", "int", mapping="custom_path")

        assert "new_field" in metadata_creator.schema
        assert metadata_creator.get_field_type("new_field") == "int"
        assert metadata_creator.get_field_mapping("new_field") == "custom_path"

        # Test default mapping (uses field_name as mapping)
        metadata_creator.add_schema_field("another_field", "str")
        assert metadata_creator.get_field_mapping("another_field") == "another_field"

    def test_extract_first_value_from_dict(self, metadata_creator):
        """Test that direct mapping extracts first value from dict"""
        data = {"test_field": {"first_key": "first_value", "second_key": "second_value"}}
        metadata_creator.register_field_mapping("test", "test_field")
        metadata = metadata_creator.create(data)
        # Should extract first value from the dict
        assert metadata["test"] == "first_value"

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

    def test_create_metadata_with_list_field(self, metadata_creator, sample_extracted_result):
        """Test that list fields are returned directly"""
        metadata = metadata_creator.create(sample_extracted_result)

        # coverage and cvg_premium are list type fields
        assert "coverage" in metadata
        assert isinstance(metadata["coverage"], list)
        assert len(metadata["coverage"]) == 1
        assert metadata["coverage"][0]["保险名称"] == "个人癌症医疗保险（互联网2022版A款）"

        assert "cvg_premium" in metadata
        assert isinstance(metadata["cvg_premium"], list)
