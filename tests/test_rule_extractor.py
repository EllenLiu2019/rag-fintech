import pytest
import json
from pathlib import Path
from service.extractor.rule_extractor import RuleExtractor

# 定义数据目录
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def policy_documents():
    """
    加载测试数据 fixture
    """
    data_file = DATA_DIR / "policy_test_data.json"
    if not data_file.exists():
        pytest.fail(f"Test data file not found: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


class TestRuleExtractor:

    def test_extract(self, policy_documents):
        schema_path = DATA_DIR / "insurance.json"

        if not schema_path.exists():
            pytest.fail(f"Test schema file not found: {schema_path}")

        rule_extractor = RuleExtractor(schema_path=str(schema_path))
        rule_extractor.extract(policy_documents)

        result = rule_extractor.extracted_result

        # 1. 验证保单号
        assert "policy_number" in result
        assert result["policy_number"]["raw_value"] == "AO1234567890FE"

        # 2. 验证投保人信息
        assert "policy_holder" in result
        holder = result["policy_holder"]
        assert holder["name"]["raw_value"] == "Ellen Liu"
        assert holder["gender"]["raw_value"] == "女"
        assert holder["birth_date"]["raw_value"] == "1890-10-13"
        assert holder["id_number"]["raw_value"] == "6234567890R"

        # 3. 验证被保险人信息
        assert "insured" in result
        insured = result["insured"]
        assert insured["name"]["raw_value"] == "Song Qihui"
        assert insured["gender"]["raw_value"] == "女"
        assert insured["birth_date"]["raw_value"] == "1973-09-20"
        assert insured["id_number"]["raw_value"] == "8234567891R"
        assert insured["relationship_to_holder"]["raw_value"] == "父母"

        # 4. 验证日期提取
        assert "effective_date" in result
        assert result["effective_date"]["raw_value"] == "2025年08月04日"

        assert "expiry_date" in result
        assert result["expiry_date"]["raw_value"] == "2026年08月03日"

        # 5. 验证保障内容 (列表表格)
        assert "coverage" in result
        coverages = result["coverage"]
        assert isinstance(coverages, list)
        assert len(coverages) == 3  # 应该有3条记录

        # 检查第一条保障
        assert coverages[0]["cvg_name"]["raw_value"] == "个人癌症医疗保险（互联网2022版A款）"
        assert coverages[0]["cvg_type"]["raw_value"] == "恶性肿瘤质子重离子医疗保险金"
        assert coverages[0]["cvg_amt"]["raw_value"] == "2,000,000"

        # 6. 验证保费明细 (列表表格)
        assert "cvg_premium" in result
        premiums = result["cvg_premium"]
        assert isinstance(premiums, list)
        assert len(premiums) == 2  # 应该有2条记录

        # 检查第一条保费
        assert premiums[0]["cvg_name"]["raw_value"] == "个人癌症医疗保险（互联网2022版A款）"
        assert premiums[0]["premium"]["raw_value"] == "2,284.00"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
