# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
"""

import re
from typing import Any
import networkx as nx
from dataclasses import dataclass

from common import get_logger
from graphrag.extractor import Extractor
from graphrag.utils import chat_limiter, split_string_by_multi_markers
from common.constants import (
    ENTITY_EXTRACTION_MAX,
    TUPLE_DELIMITER,
    RECORD_DELIMITER,
    COMPLETION_DELIMITER,
)

logger = get_logger(__name__)


@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph
    source_docs: dict[Any, Any]


class GraphExtractor(Extractor):
    """Unipartite graph extractor class definition."""

    _max_gleanings: int

    def __init__(
        self,
        max_extraction: int | None = None,
    ):
        super().__init__()
        self.max_extraction = max_extraction or ENTITY_EXTRACTION_MAX

        # Wire defaults into the prompt variables
        self.prompt_variables = {
            "tuple_delimiter": TUPLE_DELIMITER,
            "record_delimiter": RECORD_DELIMITER,
            "completion_delimiter": COMPLETION_DELIMITER,
        }

    async def process_single_content(
        self,
        clause_id: int,
        clause_text: str,
        out_results: dict[int, tuple[dict[str, Any], dict[str, Any], int]],
    ):

        variables = {
            **self.prompt_variables,
            "input_text": clause_text,
        }
        hint_prompt = self.prompt_manager.get("graph_extraction", **variables)
        async with chat_limiter:
            logger.debug(f"Acquired chat limiter, {chat_limiter._value} slots remaining")
            response, tokens = await self.chat(hint_prompt, [{"role": "user", "content": "Output:"}])

        results = response or ""
        history = [{"role": "system", "content": hint_prompt}, {"role": "user", "content": response}]

        total_tokens = tokens
        # Repeat to ensure we maximize entity count
        # for i in range(self.max_extraction):
        #     history.append({"role": "user", "content": self.prompt_manager.get("continue_prompt")})
        #     async with chat_limiter:
        #         logger.debug(f"Acquired chat limiter, {chat_limiter._value} slots remaining")
        #         response, tokens = await self.chat("", history)
        #     results += response or ""
        #     total_tokens += tokens

        #     # if this is the final glean, don't bother updating the continuation flag
        #     if i >= self.max_extraction - 1:
        #         break
        #     history.append({"role": "assistant", "content": response})
        #     history.append({"role": "user", "content": self.prompt_manager.get("loop_prompt")})
        #     async with chat_limiter:
        #         logger.debug(f"Acquired chat limiter, {chat_limiter._value} slots remaining")
        #         continuation, tokens = await self.chat("", history)
        #     total_tokens += tokens
        #     if continuation != "Y":
        #         break
        #     history.append({"role": "assistant", "content": "Y"})

        records = split_string_by_multi_markers(
            results,
            [
                self.prompt_variables["record_delimiter"],
                self.prompt_variables["completion_delimiter"],
            ],
        )
        rcds = []
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            rcds.append(record.group(1))
        records = rcds
        maybe_nodes, maybe_edges = self._entities_and_relations(records, self.prompt_variables["tuple_delimiter"])
        out_results[clause_id] = (maybe_nodes, maybe_edges, total_tokens)
        logger.info(f"Extracted {len(maybe_nodes)} nodes, {len(maybe_edges)} edges for clause {clause_id}")
