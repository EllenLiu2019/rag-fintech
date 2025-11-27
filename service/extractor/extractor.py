import logging
from typing import Any, Optional
from copy import deepcopy

from service.extractor.rule_extractor import RuleExtractor
from service.extractor.llm_extractor import LLMExtractor
from service.extractor.converter import FieldConvert
from service.utils.confidence_calculator import ConfidenceCalculator
from service.extractor.metadata_creator import MetadataCreator

logger = logging.getLogger(__name__)


class Extractor:
    """
    多策略提取流水线：规则 + 模式 + LLM

    提取策略：
    - 强格式字段（保单号、日期）→ 规则优先
    - 弱格式字段（特别约定）→ LLM
    - 关键字段 → 交叉验证
    """

    def __init__(self) -> None:
        self.llm_extractor: LLMExtractor = LLMExtractor()
        self.rule_extractor: RuleExtractor = RuleExtractor()
        self.field_convert: FieldConvert = FieldConvert()
        self.confidence_calculator: ConfidenceCalculator = ConfidenceCalculator()
        self.metadata_creator = MetadataCreator()

    def extract(
        self, documents: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """
        执行完整的提取流程

        流程步骤：
        1. 使用规则提取器提取结构化数据
        2. 对提取结果进行字段转换（如数字格式化）
        3. 计算提取结果的置信度
        4. 创建元数据
        5. 如果置信度低于阈值，可触发 LLM 提取（当前未启用）

        Args:
            documents: 包含文档数据的列表，格式为 [dict[str, Any], ...]

        Returns:
            tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: 转换后的提取结果、置信度结果和元数据结果

        Raises:
            ValueError: 当 documents 格式不正确时
            Exception: 提取过程中的其他错误
        """
        if not isinstance(documents, list) or not all(isinstance(doc, dict) for doc in documents):
            raise ValueError("documents must be a list of dict objects")

        try:
            # 步骤 1: 规则提取（快速、准确）
            logger.info("Starting rule-based extraction")
            self.rule_extractor.extract(documents)

            # 步骤 2: 字段转换（格式化数字等）
            logger.info("Converting extracted fields")
            self._convert_extracted_fields()

            # 步骤 3: 计算置信度
            logger.info("Calculating confidence scores")
            confidence_result = self.confidence_calculator.calculate(
                self.rule_extractor.extracted_result,
                self.rule_extractor.schema,
                self.rule_extractor.soups,
            )

            # 步骤 4: 低置信度时触发 LLM 提取（可选）
            if confidence_result.get("overall_confidence", 0.0) < 0.8:
                logger.warning(
                    f"Low confidence detected: {confidence_result.get('overall_confidence')}. "
                    "LLM extraction is available but currently disabled."
                )
                # TODO: 启用 LLM 提取作为补充
                # self._fallback_to_llm_extraction(document)

            extracted_data = self.rule_extractor.extracted_result
            metadata = self.metadata_creator.create(extracted_data)
            return extracted_data, confidence_result, metadata

        except Exception as e:
            logger.error(f"Extraction pipeline failed: {e}", exc_info=True)
            raise

    def _convert_extracted_fields(self) -> None:
        try:
            self.field_convert.convert(self.rule_extractor.extracted_result)
        except Exception as e:
            logger.warning(f"Field conversion failed: {e}", exc_info=True)

    def _fallback_to_llm_extraction(self, document: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            hints = deepcopy(self.rule_extractor.extracted_result)

            content = "\n".join([doc.get("text", "") for doc in document.get("documents", [])])

            llm_results = self.llm_extractor.extract(content, hints=hints)
            logger.info("LLM extraction completed")
            return llm_results

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            return None

    def get_extracted_result(self) -> dict[str, Any]:
        """
        获取当前提取结果

        Returns:
            dict[str, Any]: 提取结果字典
        """
        return self.rule_extractor.extracted_result

    def reset(self) -> None:
        """
        重置提取器状态，清空之前的提取结果
        """
        self.rule_extractor.extracted_result = {}
        self.rule_extractor.soups = []
        logger.debug("Extraction pipeline reset")


# if __name__ == "__main__":
#     path = "api/db/policy_mini.json"
#     with open(path, "r") as f:
#         document = json.load(f)
#     extractor = Extractor()
#     converted_result, confidence_result = extractor.extract(document)

#     print("=" * 60)
#     print("转换结果:")
#     print("=" * 60)
#     print(json.dumps(converted_result, ensure_ascii=False, indent=4))

#     print("\n" + "=" * 60)
#     print("置信度结果:")
#     print("=" * 60)
#     print(json.dumps(confidence_result, ensure_ascii=False, indent=4))

#     metadata_creator = MetadataCreator()
#     metadata = metadata_creator.create(converted_result)
#     print("=" * 60)
#     print("元数据:")
#     print("=" * 60)
#     print(json.dumps(metadata, ensure_ascii=False, indent=4))
