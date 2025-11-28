import re
from typing import Any, Optional


class ConfidenceCalculator:

    def calculate(self, policy_result: dict, schema: dict, soups: list[Any]) -> dict[str, Any]:
        confidence_report = {
            "overall_confidence": 0.0,
            "field_confidence": {},
            "completeness": {},
            "warnings": [],
            "errors": [],
        }

        # 1. 计算字段完整性
        completeness_scores = self.calculate_field_completeness(policy_result, schema)
        confidence_report["completeness"] = completeness_scores

        # 2. 计算每个字段的置信度
        field_confidences = {}
        soup_texts = "" if not soups else "\n==========\n".join([str(soup) for soup in soups])

        for field_key, field_value in policy_result.items():
            field_config = schema.get("fields", {}).get(field_key, {})
            field_type = field_config.get("type", "")

            if field_type == "object" and isinstance(field_value, dict):
                # 对象类型：计算每个属性的置信度
                prop_confidences = {}
                for prop_key, prop_value in field_value.items():
                    if isinstance(prop_value, dict):
                        match_strategy = prop_value.get("match", {})
                        raw_value = prop_value.get("raw_value", "")
                        data_type = prop_value.get("type", "")

                        # 匹配置信度
                        match_conf = self.calculate_match_confidence(match_strategy, raw_value, soup_texts)

                        # 数据类型验证
                        type_conf, error_msg = self.validate_data_type(data_type, raw_value)

                        # 综合置信度（加权平均）
                        prop_confidence = match_conf * 0.6 + type_conf * 0.4
                        prop_confidences[prop_key] = {
                            "confidence": prop_confidence,
                            "match_confidence": match_conf,
                            "type_confidence": type_conf,
                            "raw_value": raw_value,
                            "error": error_msg if error_msg else None,
                        }

                        if prop_confidence < 0.7:
                            confidence_report["warnings"].append(
                                f"{field_key}.{prop_key}: 置信度较低 ({prop_confidence:.2f})"
                            )
                        if error_msg:
                            confidence_report["errors"].append(f"{field_key}.{prop_key}: {error_msg}")

                field_confidences[field_key] = {
                    "confidence": (
                        sum(p["confidence"] for p in prop_confidences.values()) / len(prop_confidences)
                        if prop_confidences
                        else 0.0
                    ),
                    "properties": prop_confidences,
                }

            elif field_type == "list" and isinstance(field_value, list):
                # 列表类型：计算每条记录的置信度
                list_confidences = []
                for idx, item in enumerate(field_value):
                    if isinstance(item, dict):
                        item_confidences = {}
                        for prop_key, prop_value in item.items():
                            if isinstance(prop_value, dict):
                                match_strategy = prop_value.get("match", {})
                                raw_value = prop_value.get("raw_value", "")
                                data_type = prop_value.get("type", "")

                                match_conf = self.calculate_match_confidence(match_strategy, raw_value, soup_texts)
                                type_conf, error_msg = self.validate_data_type(
                                    data_type, raw_value, prop_value.get("transform")
                                )

                                prop_confidence = match_conf * 0.6 + type_conf * 0.4
                                item_confidences[prop_key] = {
                                    "confidence": prop_confidence,
                                    "match_confidence": match_conf,
                                    "type_confidence": type_conf,
                                    "raw_value": raw_value,
                                    "error": error_msg if error_msg else None,
                                }

                        item_avg_conf = (
                            sum(p["confidence"] for p in item_confidences.values()) / len(item_confidences)
                            if item_confidences
                            else 0.0
                        )
                        list_confidences.append(
                            {
                                "index": idx,
                                "confidence": item_avg_conf,
                                "properties": item_confidences,
                            }
                        )

                field_confidences[field_key] = {
                    "confidence": (
                        sum(c["confidence"] for c in list_confidences) / len(list_confidences)
                        if list_confidences
                        else 0.0
                    ),
                    "items": list_confidences,
                }

            else:
                # 简单类型（从content提取）
                if isinstance(field_value, dict):
                    match_strategy = field_value.get("match", {})
                    raw_value = field_value.get("raw_value", "")
                    data_type = field_value.get("type", "")

                    match_conf = self.calculate_match_confidence(match_strategy, raw_value, soup_texts)
                    type_conf, error_msg = self.validate_data_type(data_type, raw_value)

                    field_confidence = match_conf * 0.6 + type_conf * 0.4
                    field_confidences[field_key] = {
                        "confidence": field_confidence,
                        "match_confidence": match_conf,
                        "type_confidence": type_conf,
                        "raw_value": raw_value,
                        "error": error_msg if error_msg else None,
                    }

                    if field_confidence < 0.7:
                        confidence_report["warnings"].append(f"{field_key}: 置信度较低 ({field_confidence:.2f})")

        confidence_report["field_confidence"] = field_confidences

        # 3. 计算总体置信度（加权平均）
        all_confidences = []
        for field_key, field_conf in field_confidences.items():
            all_confidences.append(field_conf["confidence"])
            # 考虑完整性
            completeness = completeness_scores.get(field_key, 0.0)
            all_confidences.append(completeness)

        if all_confidences:
            confidence_report["overall_confidence"] = sum(all_confidences) / len(all_confidences)

        return confidence_report

    def calculate_match_confidence(
        self, match_strategy: dict, extracted_value: str, source_texts: list[str] = []
    ) -> float:
        strategy = match_strategy.get("strategy", "")
        confidence = 0.0

        if strategy == "exact":
            # 精确匹配：置信度最高
            values = match_strategy.get("values", [])
            if extracted_value and any(v in source_texts for v in values):
                confidence = 1.0
            elif extracted_value:
                confidence = 0.8  # 有值但无法验证匹配
            else:
                confidence = 0.0  # 无值
        elif strategy == "regex":
            # 正则匹配：根据匹配质量评分
            regex = match_strategy.get("regex", "")
            if not regex:
                return 0.0

            match = re.search(regex, source_texts)
            if match and extracted_value:
                confidence = 1.0
            elif extracted_value:
                confidence = 0.6  # 有值但正则未匹配
            else:
                confidence = 0.0
        return confidence

    def validate_data_type(
        self, field_type: str, raw_value: str, transform: Optional[dict] = None
    ) -> tuple[float, str]:
        """
        验证数据类型和格式

        Returns:
            (置信度, 错误信息)
        """
        if not raw_value or not raw_value.strip():
            return (0.0, "值为空")

        confidence = 1.0
        error_msg = ""

        if field_type == "number":
            # 验证数字格式
            cleaned = raw_value.replace(",", "").replace("¥", "").replace("元", "").strip()
            try:
                float(cleaned)
                confidence = 1.0
            except ValueError:
                confidence = 0.3
                error_msg = f"无法转换为数字: {raw_value}"

        elif field_type == "string":
            # 字符串基本验证
            if len(raw_value.strip()) == 0:
                confidence = 0.0
                error_msg = "字符串为空"
            # elif len(raw_value.strip()) < 2:
            #     confidence = 0.5
            #     error_msg = "字符串过短"
            else:
                confidence = 1.0

        elif field_type == "date":
            # 验证日期格式
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",
                r"\d{4}年\d{2}月\d{2}日",
                r"\d{4}/\d{2}/\d{2}",
            ]
            if any(re.search(pattern, raw_value) for pattern in date_patterns):
                confidence = 1.0
            else:
                confidence = 0.4
                error_msg = f"日期格式不正确: {raw_value}"

        return (confidence, error_msg)

    def calculate_table_match_confidence(table_id: str, table_mapping_strategy: dict, matched_text: str) -> float:
        """
        计算表格匹配的置信度
        """
        if not table_id:
            return 0.0

        strategy_info = table_mapping_strategy.get(table_id, {})
        strategy = strategy_info.get("strategy", "")
        values = strategy_info.get("values", [])

        if strategy == "exact" and matched_text in values:
            return 1.0
        elif strategy == "exact" and matched_text:
            # 部分匹配或相似匹配
            for v in values:
                if v in matched_text or matched_text in v:
                    return 0.7
            return 0.3
        else:
            return 0.0

    def calculate_field_completeness(self, policy_result: dict, schema: dict) -> dict[str, float]:
        """
        计算字段完整性（必需字段是否都提取到了）
        """
        completeness_scores = {}

        for field_key, field_config in schema.get("fields", {}).items():
            if field_key not in policy_result:
                completeness_scores[field_key] = 0.0
                continue

            extracted = policy_result[field_key]

            if field_config.get("type") == "object":
                # 对象类型：检查所有properties是否都提取到了
                required_props = field_config.get("properties", {}).keys()
                if isinstance(extracted, dict):
                    extracted_props = set(extracted.keys())
                    required_props_set = set(required_props)
                    if required_props_set.issubset(extracted_props):
                        completeness_scores[field_key] = 1.0
                    else:
                        missing = required_props_set - extracted_props
                        completeness_scores[field_key] = 1.0 - (len(missing) / len(required_props_set))
                else:
                    completeness_scores[field_key] = 0.0

            elif field_config.get("type") == "list":
                # 列表类型：检查是否至少有一条记录
                if isinstance(extracted, list) and len(extracted) > 0:
                    # 检查第一条记录是否包含所有必需属性
                    first_item = extracted[0]
                    required_props = field_config.get("properties", {}).keys()
                    if isinstance(first_item, dict):
                        extracted_props = set(first_item.keys())
                        required_props_set = set(required_props)
                        if required_props_set.issubset(extracted_props):
                            completeness_scores[field_key] = 1.0
                        else:
                            missing = required_props_set - extracted_props
                            completeness_scores[field_key] = 1.0 - (len(missing) / len(required_props_set))
                    else:
                        completeness_scores[field_key] = 0.5
                else:
                    completeness_scores[field_key] = 0.0

            else:
                # 简单类型：检查是否有值
                if extracted and (isinstance(extracted, dict) and extracted.get("raw_value")):
                    completeness_scores[field_key] = 1.0
                else:
                    completeness_scores[field_key] = 0.0

        return completeness_scores


def print_confidence_report(report: dict):
    """
    打印置信度报告
    """
    print("=" * 60)
    print("数据提取置信度分析报告")
    print("=" * 60)
    print(f"\n总体置信度: {report['overall_confidence']:.2%}")
    print("字段完整性:")
    for field, score in report["completeness"].items():
        status = "✓" if score == 1.0 else "⚠" if score >= 0.7 else "✗"
        print(f"  {status} {field}: {score:.2%}")

    print("\n字段置信度详情:")
    for field_key, field_info in report["field_confidence"].items():
        conf = field_info["confidence"]
        status = "✓" if conf >= 0.9 else "⚠" if conf >= 0.7 else "✗"
        print(f"\n  {status} {field_key}: {conf:.2%}")

        if "properties" in field_info:
            for prop_key, prop_info in field_info["properties"].items():
                prop_conf = prop_info["confidence"]
                print(
                    f"    - {prop_key}: {prop_conf:.2%} (匹配: {prop_info['match_confidence']:.2%}, 类型: {prop_info['type_confidence']:.2%})"
                )
                if prop_info.get("error"):
                    print(f"      错误: {prop_info['error']}")

        elif "items" in field_info:
            for item in field_info["items"]:
                print(f"    项目 {item['index']}: {item['confidence']:.2%}")
                for prop_key, prop_info in item["properties"].items():
                    prop_conf = prop_info["confidence"]
                    print(f"      - {prop_key}: {prop_conf:.2%}")

    if report["warnings"]:
        print(f"\n⚠ 警告 ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["errors"]:
        print(f"\n✗ 错误 ({len(report['errors'])}):")
        for error in report["errors"]:
            print(f"  - {error}")

    print("\n" + "=" * 60)
