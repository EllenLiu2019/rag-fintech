from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

from transformers import AutoTokenizer


@dataclass(frozen=True)
class PromptBuilderConfig:
    model_id: str = "Qwen/Qwen3-8B"
    prefix_text: str = (
        "You are a helpful assistant. Read the following text and reply with a short acknowledgment.\n" "TEXT:\n"
    )
    nonce_pool_size: int = 256
    nonce_tokens_len: int = 16
    filler_texts: tuple[str, ...] = (
        "lorem ipsum ",
        "data token ",
        "benchmark input ",
        "sequence padding ",
        "neutral text ",
        "repeat pattern ",
    )


class Qwen3PromptBuilder:
    """
    Prompt generator for vLLM benchmarking with exact input-token control.

    target_tokens refers to the FULL prompt_tokens that vLLM will report,
    including all chat-template overhead (special tokens, role markers, etc.).
    The builder uses apply_chat_template to measure the same token sequence
    that vLLM constructs internally, so the generated content length is
    adjusted to compensate for template overhead.
    """

    def __init__(self, config: PromptBuilderConfig):
        self.cfg = config
        self.tok = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)

        self.prefix_ids = self.tok.encode(self.cfg.prefix_text, add_special_tokens=False)
        if len(self.prefix_ids) == 0:
            raise ValueError("prefix_text encodes to empty token sequence; please change it.")

        self.nonce_pool: List[List[int]] = self._build_nonce_pool(
            pool_size=self.cfg.nonce_pool_size,
            nonce_len=self.cfg.nonce_tokens_len,
        )

        self.filler_ids_pool: List[List[int]] = [
            self.tok.encode(t, add_special_tokens=False) for t in self.cfg.filler_texts
        ]
        if any(len(x) == 0 for x in self.filler_ids_pool):
            raise ValueError("Some filler_texts encode to empty; please adjust filler_texts.")

        self._template_overhead = self._measure_template_overhead()

    def _measure_template_overhead(self) -> int:
        """Measure how many tokens the chat template adds around user content."""
        probe = "x"
        probe_content_tokens = len(self.tok.encode(probe, add_special_tokens=False))
        full_tokens = len(
            self.tok.apply_chat_template(
                [{"role": "user", "content": probe}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        return full_tokens - probe_content_tokens

    def _content_tokens(self, content: str) -> int:
        return len(self.tok.encode(content, add_special_tokens=False))

    def _full_prompt_tokens(self, content: str) -> int:
        """Equivalent to vLLM's prompt_tokens: content tokens + chat template overhead."""
        return self._content_tokens(content) + self._template_overhead

    def build(self, target_tokens: int, seed: int = 0) -> str:
        """
        Build a prompt whose vLLM prompt_tokens equals EXACTLY target_tokens.

        The returned string is the user-message content; the caller sends it via
        the OpenAI chat-completions API as messages=[{"role":"user","content": ...}].
        """
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive.")

        content_target = target_tokens - self._template_overhead
        if content_target <= 0:
            raise ValueError(
                f"target_tokens ({target_tokens}) too small; "
                f"chat template alone uses {self._template_overhead} tokens."
            )

        nonce_text = self.tok.decode(
            self.nonce_pool[seed % len(self.nonce_pool)],
            skip_special_tokens=False,
        )

        text = self.cfg.prefix_text + nonce_text

        rng = random.Random(seed)
        filler_strs = list(self.cfg.filler_texts)

        while self._content_tokens(text) < content_target:
            text += filler_strs[rng.randrange(len(filler_strs))]

        cur = self._content_tokens(text)
        if cur > content_target:
            content_ids = self.tok.encode(text, add_special_tokens=False)[:content_target]
            text = self.tok.decode(content_ids, skip_special_tokens=False)

            # BPE decode→re-encode can shift count; converge with a trim loop
            while self._content_tokens(text) > content_target:
                diff = self._content_tokens(text) - content_target
                content_ids = self.tok.encode(text, add_special_tokens=False)
                content_ids = content_ids[: len(content_ids) - max(diff, 1)]
                text = self.tok.decode(content_ids, skip_special_tokens=False)

            while self._content_tokens(text) < content_target:
                text += filler_strs[0]
                if self._content_tokens(text) > content_target:
                    overshoot = self._content_tokens(text) - content_target
                    content_ids = self.tok.encode(text, add_special_tokens=False)
                    content_ids = content_ids[: len(content_ids) - overshoot]
                    text = self.tok.decode(content_ids, skip_special_tokens=False)
                    break

        return text

    def count_tokens(self, text: str) -> int:
        """Count the full prompt_tokens as vLLM would report them."""
        return self._full_prompt_tokens(text)

    def _build_nonce_pool(self, pool_size: int, nonce_len: int) -> List[List[int]]:
        nonce_fragments = [
            " nonceA ", " nonceB ", " nonceC ", " nonceD ",
            " idXX ", " tagYY ", " seedZZ ", " keyQQ ",
            " alpha ", " beta ", " gamma ", " delta ",
        ]
        frag_ids = [self.tok.encode(s, add_special_tokens=False) for s in nonce_fragments]
        frag_ids = [x for x in frag_ids if len(x) > 0]
        if not frag_ids:
            raise ValueError("Failed to build nonce fragments; tokenizer produced empty encodings.")

        pool: List[List[int]] = []
        for i in range(pool_size):
            rng = random.Random(i * 9973 + 17)
            ids: List[int] = []
            while len(ids) < nonce_len:
                frag = frag_ids[rng.randrange(len(frag_ids))]
                need = nonce_len - len(ids)
                if len(frag) <= need:
                    ids.extend(frag)
                else:
                    ids.extend(frag[:need])
            pool.append(ids[:nonce_len])
        return pool


# --------- Example usage ----------
if __name__ == "__main__":
    cfg = PromptBuilderConfig(model_id="Qwen/Qwen3-8B", nonce_tokens_len=16)
    builder = Qwen3PromptBuilder(cfg)
    print(f"Chat template overhead: {builder._template_overhead} tokens")

    for t in [1024, 2048, 4096, 8192]:
        p = builder.build(target_tokens=t, seed=1)
        full = builder.count_tokens(p)
        raw = len(builder.tok.encode(p, add_special_tokens=False))
        print(f"target={t}  full_prompt_tokens={full}  content_tokens={raw}  overhead={full - raw}")
