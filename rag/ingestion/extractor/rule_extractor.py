import re
import json
from bs4 import BeautifulSoup, Tag
from typing import Optional, Any
from rag.ingestion.extractor.html_table_grid import HtmlTableGrid
from common import get_logger

logger = get_logger(__name__)

HEADER_IDENTIFIER = "th"
ROW_IDENTIFIER = "tr"
CELL_IDENTIFIER = "td"


class RuleExtractor:
    """
    extractor based on schema matching, including exact matching and regex matching
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        self.table_mapping_strategy: dict[str, dict[str, Any]] = {}
        self.content_mapping_strategy: dict[str, dict[str, Any]] = {}
        self.schema: dict[str, Any] = schema
        for key, config in self.schema.get("fields", {}).items():
            if config.get("position") == "table":
                self.table_mapping_strategy[key] = config
            elif config.get("position") == "content":
                self.content_mapping_strategy[key] = dict(config.get("match", {}))
        self.ignore_keywords = frozenset(self.schema.get("ignore_keywords", []))

        self.grids: list[HtmlTableGrid] = []
        self.extracted_results: dict[str, Any] = {}
        # for confidence calculation
        self.extraction_signals: dict[str, dict[str, Any]] = {}

    def extract(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        for document in documents:
            if self._is_extracted_finished():
                logger.info("Extraction finished")
                return self.extracted_results
            soup = BeautifulSoup(document["text"], "html.parser")
            self.content_extract(soup)
            tables = soup.find_all("table")
            self.identify_tables(tables)
            self.extract_identified_tables(tables)
            self.extract_from_grids()
        return self.extracted_results

    def _is_extracted_finished(self) -> bool:
        return len(self.extracted_results.keys()) == len(self.schema.get("fields", {}))

    def content_extract(self, soup: BeautifulSoup) -> None:
        for key, item in self.content_mapping_strategy.items():
            # Skip if already successfully matched (avoid overwriting with failed match)
            if key in self.extraction_signals and self.extraction_signals[key].get("matched"):
                continue

            match = re.search(item["regex"], str(soup))
            if not match:
                # Only record failed match signal if not already recorded
                if key not in self.extraction_signals:
                    self.extraction_signals[key] = {
                        "source": "content_regex",
                        "matched": False,
                        "strategy": "regex",
                    }
                continue

            key_name = item.get("key_name")
            value = match.group(item["group"])
            self.extracted_results[key] = {key_name: value}
            # Record successful match signal
            self.extraction_signals[key] = {
                "source": "content_regex",
                "matched": True,
                "strategy": "regex",
                "match_span": match.span(),
                "full_match": match.group(0),
            }

    def identify_tables(self, tables: list[Tag]) -> None:
        for table in tables:
            # find matched key and strategy from header
            self._identify_table_with_header(table)

            # fallback to build grid if header identification failed
            if not table.get("id"):
                self._fallback_build_grid(table)

    def _identify_table_with_header(self, table: Tag) -> None:
        header_cells = table.find_all(HEADER_IDENTIFIER)
        if not header_cells:
            return

        header_text = header_cells[0].text.strip()

        # check if the first table header cell matches the strategy
        matched_key, matched_strategy = self._match_table_strategy(header_text)

        if matched_key and matched_strategy:
            table_type = matched_strategy["match"].get("type")

            if table_type != "list":
                return

            table["id"] = matched_key
            logger.info(f"Identified table with thead: {matched_key}")
            # Record table header match signal
            self.extraction_signals[matched_key] = {
                "source": "table_header",
                "matched": True,
                "strategy": "exact",
                "matched_text": header_text,
                "table_type": table_type,
            }

    def _fallback_build_grid(self, table: Tag) -> None:
        html_table_grid = HtmlTableGrid(table=table, match_strategy=self._match_table_strategy)
        self.grids.append(html_table_grid)

    def extract_identified_tables(self, tables: list[Tag]) -> None:
        for table in tables:
            table_id = table.get("id")

            if not table_id:
                logger.warning("Table missing id or title, skipping")
                continue

            # Normalize table_id: if it's a list, take the first element
            if isinstance(table_id, list):
                table_id = table_id[0] if table_id else None
                if not table_id:
                    logger.warning("Table has empty id list, skipping")
                    continue

            try:
                self._extract_list_table(table, table_id)
            except Exception as e:
                logger.error(f"Failed to extract table {table_id}: {e}", exc_info=True)

    def extract_from_grids(self) -> None:
        for grid in self.grids:
            logger.info(f"Extracting object table with id: {grid.table_ids}")

            for tid in grid.table_ids:
                if tid not in self.extracted_results:
                    self.extracted_results[tid] = {}
                # Record grid fallback signal
                if tid not in self.extraction_signals:
                    self.extraction_signals[tid] = {
                        "source": "grid_fallback",
                        "matched": True,
                        "strategy": "exact",
                        "table_type": "object",
                        "row_count": 0,
                        "kv_count": 0,
                    }

            for row in grid.grid:
                # Skip empty rows
                if not row:
                    continue

                valid_row_id = next(
                    (
                        item.get("key")
                        for item in row
                        if item is not None and item.get("text") not in self.ignore_keywords
                    ),
                    None,
                )
                if not valid_row_id:
                    continue

                kv_count = self._extract_kv_pair(row, valid_row_id)
                # Update signal with extraction stats
                if valid_row_id in self.extraction_signals:
                    self.extraction_signals[valid_row_id]["row_count"] += 1
                    self.extraction_signals[valid_row_id]["kv_count"] += kv_count

        self.grids.clear()

    def _extract_kv_pair(self, row: list, row_id: str) -> int:
        """
        Extract key-value pairs from a row.
        Returns the number of KV pairs extracted.
        """
        if len(row) < 2:
            return 0

        # Filter out ignored keywords first
        cells_to_extract = [cell for cell in row if cell is not None and cell["text"] not in self.ignore_keywords]

        # Process as key-value pairs
        kv_count = 0
        iterator = iter(cells_to_extract)
        for key_cell in iterator:
            value_cell = next(iterator, None)
            if value_cell:
                self.extracted_results[row_id][key_cell["text"]] = value_cell["text"]
                kv_count += 1
        return kv_count

    def _extract_list_table(self, table: Tag, table_id: str) -> None:
        """
        extract list type table (multi-row data format, with table header)
        """
        logger.info(f"Extracting list table with id: {table_id}")

        header_cells = table.find_all(HEADER_IDENTIFIER)
        headers = [cell.text.strip() for cell in header_cells]

        rows = table.find_all(ROW_IDENTIFIER)
        self.extracted_results[table_id] = []

        for row in rows:
            row_data = self._extract_row_by_header(row, headers)
            if row_data:
                self.extracted_results[table_id].append(row_data)

    def _extract_row_by_header(self, row: Tag, headers: list[str]) -> dict[str, Any]:
        """
        extract one row data from list type table
        """
        row_data = {}
        cells = row.find_all(CELL_IDENTIFIER)

        for col_index, cell in enumerate(cells):
            if col_index >= len(headers):
                continue

            header_text = headers[col_index]

            row_data[header_text] = cell.text.strip()
        return row_data

    def _match_table_strategy(self, text: str, cell_index: int = 0) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        for key, strategy in self.table_mapping_strategy.items():
            if self._is_strategy_match(strategy, text, cell_index):
                return key, strategy
        return None, None

    def _is_strategy_match(self, strategy: dict[str, Any], text: str, cell_index: int) -> bool:
        match_strategy = strategy.get("match", {})
        target = match_strategy.get("target")
        values = match_strategy.get("values", [])

        if target == cell_index and text in values:
            return True

        return False

    def clear(self) -> None:
        self.extracted_results.clear()
        self.extraction_signals.clear()
        self.grids.clear()
