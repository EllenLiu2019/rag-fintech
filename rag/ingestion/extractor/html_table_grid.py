from bs4 import Tag

from common import get_logger

logger = get_logger(__name__)


class HtmlTableGrid:
    def __init__(self, table: Tag, fallback_strategy: callable) -> None:
        self.fallback_strategy = fallback_strategy
        self.table = table
        self.grid = []
        self.table_ids = []
        self.extracted = False
        self.build_grid()

    def build_grid(self) -> None:

        row_id = ""
        for row_idx, tr in enumerate(self.table.find_all("tr")):

            # find all cells of the row, which may be td or th
            for cell_idx, cell in enumerate(tr.find_all(["th", "td"])):

                text = cell.get_text(strip=True)

                # identify the 1st cell of the row to get the matched key and strategy
                # only match the 1st cell of each row with matched_keys, if matched, the row and subsequent rows belong to the matched key
                if cell_idx == 0:
                    matched_key, matched_strategy = self.fallback_strategy(text)
                    if matched_key:
                        if matched_key not in self.table_ids:
                            self.table_ids.append(matched_key)
                        row_id = matched_key
                        logger.info(f"Identified 1st cell {text} belongs to: {row_id}")
                    elif row_id:
                        logger.info(f"Assumed 1st cell {text} belongs to: {row_id}")

                # if the 1st cell of the row is not matched with any of the matched_keys,
                # but the previous row has been matched,
                # then the row belongs to the previous row's key
                # if every row is not matched with any of the matched_keys,
                # then no information is saved,
                # and the subsequent information extraction is skipped
                if row_id:
                    # skip cells occupied by rowspan from above
                    self._ensure_cell(row_idx, cell_idx)
                    while self._is_occupied(row_idx, cell_idx):
                        cell_idx += 1
                        self._ensure_cell(row_idx, cell_idx)

                    # get spans
                    rowspan = int(cell.get("rowspan", 1))
                    colspan = int(cell.get("colspan", 1))

                    for i in range(rowspan):
                        for j in range(colspan):
                            target_row_idx = row_idx + i
                            target_col_idx = cell_idx + j
                            self._ensure_cell(target_row_idx, target_col_idx)
                            test_to_use = text
                            if rowspan > 1 or (colspan > 1 and j > 0):
                                test_to_use = "SPAN"

                            self.grid[target_row_idx][target_col_idx] = {
                                "text": test_to_use,
                                "key": row_id,
                            }

                    cell_idx += colspan

    def _ensure_cell(self, row_idx: int, col_idx: int) -> None:
        while len(self.grid) <= row_idx:
            self.grid.append([])
        while len(self.grid[row_idx]) <= col_idx:
            self.grid[row_idx].append(None)

    def _is_occupied(self, row_idx: int, col_idx: int) -> bool:
        if row_idx >= len(self.grid) or col_idx >= len(self.grid[row_idx]):
            return False
        return self.grid[row_idx][col_idx] is not None
