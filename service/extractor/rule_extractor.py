import re
import json
import logging
from bs4 import BeautifulSoup, Tag
from typing import Optional, Any, Union

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class RuleExtractor:
    """
    基于 schema 匹配的提取器，包含 精确匹配 和 正则匹配
    """

    def __init__(self, schema_path: str = "service/extractor/schema/insurance.json") -> None:
        self.table_mapping_strategy: dict[str, dict[str, Any]] = {}
        self.extract_from_table: dict[str, dict[str, Any]] = {}
        self.extract_from_content: dict[str, dict[str, Any]] = {}
        self.extracted_result: dict[str, Any] = {}
        self.soups: list[BeautifulSoup] = []
        with open(schema_path, "r") as f:
            self.schema: dict[str, Any] = json.load(f)
            self.parse_schema(self.schema)

    def extract(self, documents: list[dict[str, Any]]) -> None:
        for document in documents:
            soup = BeautifulSoup(document["text"], "html.parser")
            self.soups.append(soup)
            self.content_extract(soup)
            tables = soup.find_all("table")
            self.table_identify(tables)
            self.table_extract(tables)

    def parse_schema(self, schema: dict[str, Any]) -> None:
        def table_schema_item(key: str, config: dict[str, Any]) -> None:
            prop_list: list[dict[str, Any]] = []
            for prop_name, prop_config in config.get("properties", {}).items():
                if prop_config.get("position") == "table":
                    table_schema_item(prop_name, prop_config)
                else:
                    prop_list.append(
                        {
                            "key": prop_name,
                            "type": prop_config.get("type"),
                            "match": dict(prop_config.get("match", {})),
                            "transform": prop_config.get("transform"),
                        }
                    )
            schema_item = {
                "type": config.get("type"),
                "match": dict(config.get("match", {})),
                "properties": prop_list,
            }
            self.extract_from_table[key] = schema_item
            self.table_mapping_strategy[key] = {
                "strategy": config.get("match", {}).get("strategy"),
                "values": config.get("match", {}).get("values"),
                "target": config.get("match", {}).get("target"),
                "table_type": config.get("type"),
            }

        for key, config in schema.get("fields", {}).items():
            if config.get("position") == "table":
                table_schema_item(key, config)
            elif config.get("position") == "content":
                schema_item = {
                    "type": config.get("type"),
                    "match": dict(config.get("match", {})),
                }
                self.extract_from_content[key] = schema_item

    def content_extract(self, soup: BeautifulSoup) -> None:
        for key, item in self.extract_from_content.items():
            match_dict = item.get("match")
            match = re.search(match_dict.get("regex"), str(soup))
            if not match:
                continue
            value = match.group(match_dict.get("group"))
            match_text = {
                "type": item.get("type"),
                "match": match_dict,
                "raw_value": value,
            }
            self.extracted_result[key] = match_text

    def table_identify(self, tables: list[Tag]) -> None:
        """
        识别表格类型并设置表格和行的标识符

        Args:
            tables: BeautifulSoup 表格元素列表
        """
        for table in tables:
            thead = table.find("thead")

            if thead:
                self._identify_table_with_thead(table, thead)
            else:
                self._identify_table_without_thead(table)

    def _identify_table_with_thead(self, table: Tag, thead: Tag) -> None:
        """
        识别有 thead 的表格（列表类型表格）

        Args:
            table: BeautifulSoup 表格元素
            thead: BeautifulSoup thead 元素
        """
        header_cells = thead.find_all("th")
        if not header_cells:
            return

        # 检查第一个表头单元格是否匹配策略
        first_header = header_cells[0]
        header_text = first_header.text.strip()

        matched_key, matched_strategy = self._match_table_strategy(header_text, cell_index=0)

        if matched_key and matched_strategy:
            table["id"] = matched_key
            table["title"] = matched_strategy["table_type"]
            logger.debug(
                f"Identified table with thead: {matched_key} ({matched_strategy['table_type']})"
            )

    def _identify_table_without_thead(self, table: Tag) -> None:
        """
        识别没有 thead 的表格（对象类型表格，可能有 rowspan）

        Args:
            table: BeautifulSoup 表格元素
        """
        rows = table.find_all("tr")
        if not rows:
            return

        remaining_rowspan = -1
        current_row_id = ""
        table_ids = []

        for row in rows:
            first_cell = row.find("td")
            if not first_cell:
                continue

            rowspan_value = first_cell.get("rowspan")

            # 如果存在 rowspan，开始新的匹配组
            if rowspan_value:
                remaining_rowspan = int(rowspan_value) - 1
                first_cell_text = first_cell.text.strip()

                matched_key, matched_strategy = self._match_table_strategy(
                    first_cell_text, cell_index=0
                )

                if matched_key and matched_strategy:
                    current_row_id = matched_key
                    if matched_key not in table_ids:
                        table_ids.append(matched_key)
                    table["id"] = table_ids
                    table["title"] = matched_strategy["table_type"]
                    row["id"] = current_row_id
                    logger.debug(f"Identified row with rowspan: {current_row_id}")

            # 如果还在 rowspan 范围内，使用相同的 row_id
            elif remaining_rowspan > 0:
                row["id"] = current_row_id
                remaining_rowspan -= 1

    def _match_table_strategy(
        self, text: str, cell_index: int = 0
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """
        根据文本和单元格索引匹配表格策略

        Args:
            text: 单元格文本
            cell_index: 单元格索引（0 表示第一个单元格）

        Returns:
            tuple[Optional[str], Optional[dict]]: (匹配的 key, 匹配的策略) 或 (None, None)
        """
        for key, strategy in self.table_mapping_strategy.items():
            if self._is_strategy_match(strategy, text, cell_index):
                return key, strategy
        return None, None

    def _is_strategy_match(self, strategy: dict[str, Any], text: str, cell_index: int) -> bool:
        """
        检查文本是否匹配策略

        Args:
            strategy: 匹配策略字典
            text: 要匹配的文本
            cell_index: 单元格索引

        Returns:
            bool: 是否匹配
        """
        target = strategy.get("target")
        values = strategy.get("values", [])

        # 只匹配第一个单元格且文本在值列表中
        if target == "first_cell" and cell_index == 0 and text in values:
            return True

        return False

    def table_extract(self, tables: list[Tag]) -> None:
        """
        提取表格数据

        Args:
            tables: BeautifulSoup 表格元素列表
        """
        for table in tables:
            table_id = table.get("id")
            table_type = table.get("title")

            if not table_id or not table_type:
                logger.warning("Table missing id or title, skipping")
                continue

            try:
                if table_type == "object":
                    self._extract_object_table(table, table_id)
                elif table_type == "list":
                    self._extract_list_table(table, table_id)
                else:
                    logger.warning(f"Unknown table type: {table_type} for table id: {table_id}")
            except Exception as e:
                logger.error(
                    f"Failed to extract table {table_id} ({table_type}): {e}",
                    exc_info=True,
                )

    def _extract_object_table(self, table: Tag, table_id: Union[str, list[str]]) -> None:
        """
        提取对象类型表格（单行键值对格式）

        Args:
            table: BeautifulSoup 表格元素
            table_id: 表格标识符
        """
        logger.debug(f"Extracting object table with id: {table_id}")

        # 处理 table_id 可能是列表的情况
        if isinstance(table_id, list):
            table_id = table_id[0]

        self.extracted_result[table_id] = {}
        rows = table.find_all("tr")

        for row in rows:
            row_id = row.get("id")
            if not row_id:
                continue

            config_dict = self.extract_from_table.get(row_id)
            if not config_dict:
                continue

            config_prop_list = config_dict.get("properties", [])
            if not config_prop_list:
                continue

            self._extract_kv_pair(row, row_id, config_prop_list)

    def _extract_kv_pair(
        self, row: Tag, row_id: str, config_prop_list: list[dict[str, Any]]
    ) -> None:
        """
        表格中的一行数据按键值对存储

        Args:
            row: BeautifulSoup 行元素
            row_id: 行标识符
            config_prop_list: 属性配置列表
        """
        # 获取所有列，如果存在 rowspan，则跳过第一列
        col_cells = row.find_all("td")
        if col_cells and col_cells[0].get("rowspan"):
            col_cells = col_cells[1:]

        # 按对处理列（键值对）
        for i in range(0, len(col_cells) - 1, 2):
            header_cell = col_cells[i]
            value_cell = col_cells[i + 1]

            # 查找匹配的配置属性
            matched_prop = self._find_matching_property(header_cell.text.strip(), config_prop_list)

            if matched_prop:
                self._extract_matched_property(
                    value=value_cell.text.strip(),
                    matched_prop=matched_prop,
                    result_dict=self.extracted_result[row_id],
                    key=matched_prop.get("key"),
                )

    def _extract_list_table(self, table: Tag, table_id: str) -> None:
        """
        提取列表类型表格（多行数据格式，有表头）

        Args:
            table: BeautifulSoup 表格元素
            table_id: 表格标识符
        """
        logger.debug(f"Extracting list table with id: {table_id}")

        config_dict = self.extract_from_table.get(table_id)
        if not config_dict:
            logger.warning(f"No config found for table id: {table_id}")
            return

        thead = table.find("thead")
        tbody = table.find("tbody")

        if not thead or not tbody:
            logger.warning(f"Table {table_id} missing thead or tbody")
            return

        config_prop_list = config_dict.get("properties", [])
        if not config_prop_list:
            logger.warning(f"No properties config for table id: {table_id}")
            return

        # 提取表头
        header_cells = thead.find_all("th")
        headers = [cell.text.strip() for cell in header_cells]

        # 提取数据行
        rows = tbody.find_all("tr")
        self.extracted_result[table_id] = []

        for row in rows:
            row_data = self._extract_list_row(row, headers, config_prop_list)
            if row_data:
                self.extracted_result[table_id].append(row_data)

    def _extract_list_row(
        self, row: Tag, headers: list[str], config_prop_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        提取列表表格中的一行数据

        Args:
            row: BeautifulSoup 行元素
            headers: 表头列表
            config_prop_list: 属性配置列表

        Returns:
            dict: 提取的行数据
        """
        row_data = {}
        cells = row.find_all("td")

        for col_index, cell in enumerate(cells):
            if col_index >= len(headers):
                continue

            header_text = headers[col_index]
            matched_prop = self._find_matching_property(header_text, config_prop_list)

            if matched_prop:
                self._extract_matched_property(
                    value=cell.text.strip(),
                    matched_prop=matched_prop,
                    result_dict=row_data,
                    key=matched_prop.get("key"),
                )

        return row_data

    def _extract_matched_property(
        self,
        value: str,
        matched_prop: dict[str, Any],
        result_dict: Union[dict[str, Any], list[dict[str, Any]]],
        key: str,
    ) -> None:
        """
        提取信息

        Args:
            value: 值
            matched_prop: 匹配的属性
            result_dict: 结果字典
            key: 键
        """
        if isinstance(result_dict, dict):
            result_dict[key] = {
                "type": matched_prop.get("type"),
                "match": matched_prop.get("match"),
                "transform": matched_prop.get("transform"),
                "raw_value": value,
            }
        elif isinstance(result_dict, list):
            result_dict.append(
                {
                    "type": matched_prop.get("type"),
                    "match": matched_prop.get("match"),
                    "transform": matched_prop.get("transform"),
                    "raw_value": value,
                }
            )

    def _find_matching_property(
        self, text: str, config_prop_list: list[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """
        根据文本查找匹配的属性配置

        Args:
            text: 要匹配的文本
            config_prop_list: 属性配置列表

        Returns:
            dict | None: 匹配的属性配置，如果没有匹配则返回 None
        """
        for config_prop in config_prop_list:
            match_strategy = config_prop.get("match", {})
            strategy = match_strategy.get("strategy")
            values = match_strategy.get("values", [])

            if strategy == "exact" and text in values:
                return config_prop

        return None
