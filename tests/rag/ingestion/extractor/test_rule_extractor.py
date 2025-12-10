import pytest
import json
from pathlib import Path
from rag.ingestion.extractor.rule_extractor import RuleExtractor

# 定义数据目录（指向 tests/data，因为数据文件在根目录）
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


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

    @pytest.fixture
    def policy_documents_with_thead(self):
        """
        加载带 thead 样式的测试数据 fixture
        """
        data_file = DATA_DIR / "policy_html_pattern_theard.json"
        if not data_file.exists():
            pytest.fail(f"Test data file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def policy_documents_kv_pair(self):
        """
        加载 vertical KV 样式的测试数据 fixture
        """
        data_file = DATA_DIR / "policy_html_pattern_kv_pair.json"
        if not data_file.exists():
            pytest.fail(f"Test data file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def html_span_without_thead(self):
        data_file = DATA_DIR / "html_span_without_thead.json"
        if not data_file.exists():
            pytest.fail(f"Test data file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_extract_html_span_without_thead(self, html_span_without_thead):
        """
        测试 span 样式的表格提取（无 thead）
        """
        schema_path = DATA_DIR / "insurance.json"
        if not schema_path.exists():
            pytest.fail(f"Test schema file not found: {schema_path}")

        rule_extractor = RuleExtractor(schema_path=str(schema_path))
        rule_extractor.extract(html_span_without_thead)

        result = rule_extractor.extracted_result

        assert "cvg_premium" in result
        premiums = result["cvg_premium"]
        assert isinstance(premiums, list)
        assert len(premiums) == 2

        # List table extraction uses Chinese column headers as keys
        assert premiums[0]["条款名称"] == "个人癌症医疗保险（互联网2022版A款）"
        assert premiums[0]["保险费（元）"] == "2,284.00"

    def test_extract_with_kv_pair_pattern(self, policy_documents_kv_pair):
        """
        测试垂直 KV 样式的表格提取（无 thead，通过内容匹配）
        """
        schema_path = DATA_DIR / "insurance.json"
        if not schema_path.exists():
            pytest.fail(f"Test schema file not found: {schema_path}")

        rule_extractor = RuleExtractor(schema_path=str(schema_path))
        rule_extractor.extract(policy_documents_kv_pair)

        result = rule_extractor.extracted_result

        # 1. 验证保单号
        assert "policy_number" in result
        assert result["policy_number"]["保险单号"] == "AO1234567890FE"
        # 1. 验证投保人信息
        assert "policy_holder" in result
        holder = result["policy_holder"]
        assert holder["投保人"] == "Ellen Liu"
        assert holder["性别"] == "女"
        assert holder["出生日期"] == "1890-10-13"
        assert holder["证件号码"] == "6234567890R"
        assert holder["手机号码"] == "13069202805"
        assert holder["电子邮箱"] == "EllenLiu@123.com"

        # 2. 验证被保险人信息
        assert "insured" in result
        insured = result["insured"]
        assert insured["被保险人"] == "Song Qihui"
        assert insured["性别"] == "女"
        assert insured["出生日期"] == "1973-09-20"
        assert insured["证件号码"] == "8234567891R"
        assert insured["与投保人关系"] == "父母"

        # 3. 验证保障内容 (列表表格)
        assert "coverage" in result
        coverages = result["coverage"]
        assert isinstance(coverages, list)
        assert len(coverages) == 3  # 应该有3条记录

        # 检查第一条保障
        assert coverages[0]["保险名称"] == "个人癌症医疗保险（互联网2022版A款）"
        assert coverages[0]["保险责任"] == "恶性肿瘤质子重离子医疗保险金"
        assert coverages[0]["最高保险金额（元）"] == "2,000,000"
        assert (
            coverages[0]["详细说明"]
            == "首次投保或非连续投保等待期:90天免赔额:0元/年社保目录内医疗费用赔付比例:100%社保目录外医疗费用赔付比例:100%"
        )
        # 4. 验证保费明细 (列表表格)
        assert "cvg_premium" in result
        premiums = result["cvg_premium"]
        assert isinstance(premiums, list)
        assert len(premiums) == 2  # 应该有2条记录

        # 检查第一条保费
        assert premiums[0]["条款名称"] == "个人癌症医疗保险（互联网2022版A款）"
        assert premiums[0]["保险费（元）"] == "2,284.00"

    def test_extract_with_thead_pattern(self, policy_documents_with_thead):
        """
        测试带有 thead 样式的表格提取（包含 Field Type 列的情况）
        Grid fallback 处理 KV pair 格式的表格
        """
        schema_path = DATA_DIR / "insurance.json"
        if not schema_path.exists():
            pytest.fail(f"Test schema file not found: {schema_path}")

        rule_extractor = RuleExtractor(schema_path=str(schema_path))
        rule_extractor.extract(policy_documents_with_thead)

        result = rule_extractor.extracted_result

        # 1. 验证保单号 (content regex extraction)
        assert "policy_number" in result
        assert result["policy_number"]["保险单号"] == "AO1234567890FE"

        # 2. 验证投保人信息 (grid fallback KV pair extraction)
        assert "policy_holder" in result
        holder = result["policy_holder"]
        assert holder["投保人"] == "Ellen Liu"
        assert holder["性别"] == "女"
        assert holder["出生日期"] == "1890-10-13"
        assert holder["证件号码"] == "6234567890R"

        # 3. 验证被保险人信息 (grid fallback KV pair extraction)
        assert "insured" in result
        insured = result["insured"]
        assert insured["被保险人"] == "Song Qihui"
        assert insured["性别"] == "女"
        assert insured["出生日期"] == "1973-09-20"
        assert insured["证件号码"] == "8234567891R"
        assert insured["与投保人关系"] == "父母"

        # 4. 验证日期提取 (content regex extraction)
        assert "effective_date" in result
        assert result["effective_date"]["保险期间开始日期"] == "2025年08月04日"

        assert "expiry_date" in result
        assert result["expiry_date"]["保险期间结束日期"] == "2026年08月03日"

    def test_extract(self, policy_documents):
        """
        测试基于 policy_test_data.json 的复杂表格提取
        注意：该测试数据使用 rowspan 的混合 th/td 表格，policy_holder/insured 被识别为 list table
        """
        schema_path = DATA_DIR / "insurance.json"

        if not schema_path.exists():
            pytest.fail(f"Test schema file not found: {schema_path}")

        rule_extractor = RuleExtractor(schema_path=str(schema_path))
        rule_extractor.extract(policy_documents)

        result = rule_extractor.extracted_result

        # 1. 验证保单号 (content regex extraction)
        assert "policy_number" in result
        assert result["policy_number"]["保险单号"] == "AO1234567890FE"

        # 2. 验证投保人信息 - 复杂 rowspan 表格被识别为 list table
        assert "policy_holder" in result
        # 由于表格结构复杂（带 rowspan 的混合 th/td），提取为 list 格式
        assert isinstance(result["policy_holder"], list)

        # 3. 验证被保险人信息 - 同上
        assert "insured" in result
        assert isinstance(result["insured"], list)

        # 4. 验证日期提取 (content regex extraction)
        assert "effective_date" in result
        assert result["effective_date"]["保险期间开始日期"] == "2025年08月04日"

        assert "expiry_date" in result
        assert result["expiry_date"]["保险期间结束日期"] == "2026年08月03日"

        # 5. 验证保障内容 (列表表格，使用中文表头作为 key)
        assert "coverage" in result
        coverages = result["coverage"]
        assert isinstance(coverages, list)
        assert len(coverages) == 3  # 应该有3条记录

        # 检查第一条保障 - 使用中文表头作为 key
        assert coverages[0]["保险名称"] == "个人癌症医疗保险（互联网2022版A款）"
        assert coverages[0]["保险责任"] == "恶性肿瘤质子重离子医疗保险金"
        assert coverages[0]["最高保险金额（元）"] == "2,000,000"

        # 6. 验证保费明细 (列表表格)
        assert "cvg_premium" in result
        premiums = result["cvg_premium"]
        assert isinstance(premiums, list)
        assert len(premiums) == 2  # 应该有2条记录

        # 检查第一条保费 - 使用中文表头作为 key
        assert premiums[0]["条款名称"] == "个人癌症医疗保险（互联网2022版A款）"
        assert premiums[0]["保险费（元）"] == "2,284.00"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
