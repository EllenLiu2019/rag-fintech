from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

from transformers import AutoTokenizer


@dataclass(frozen=True)
class PromptBuilderConfig:
    model_id: str = "Qwen/Qwen3-8B"
    # 固定前缀：尽量短、无强语义，避免触发工具/解析路径
    prefix_text: str = (
        "You are a helpful assistant. Read the following text and reply with a short acknowledgment.\n" "TEXT:\n"
    )
    # nonce 候选池规模：越大越不容易撞
    nonce_pool_size: int = 256
    # nonce 注入的 token 长度：越靠前越能打散 prefix caching
    nonce_tokens_len: int = 16
    # 填充片段：用较“无语义”的片段，减少内容因素
    filler_texts: tuple[str, ...] = (
        "lorem ipsum ",
        "data token ",
        "benchmark input ",
        "sequence padding ",
        "neutral text ",
        "repeat pattern ",
    )
    # 如果你要兼容某些系统可能自动加 BOS，可预留 1 个 token
    reserve_for_bos: int = 0


class Qwen3PromptBuilder:
    """
    Professional prompt generator for vLLM benchmarking:
    - exact token length control with Qwen3 tokenizer
    - deterministic per (target_tokens, seed)
    - avoids prefix caching by injecting a fixed-length nonce token sequence early
    """

    def __init__(self, config: PromptBuilderConfig):
        self.cfg = config
        self.tok = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)

        # 预编码固定前缀（不加 special tokens）
        self.prefix_ids = self.tok.encode(self.cfg.prefix_text, add_special_tokens=False)
        if len(self.prefix_ids) == 0:
            raise ValueError("prefix_text encodes to empty token sequence; please change it.")

        # 构建 nonce token 序列池：每个 nonce 都是“固定 token 长度”的 token-id 序列
        # 关键点：我们直接从若干短字符串里挑，直到满足固定长度，确保每个 nonce 的 token 数一致。
        self.nonce_pool: List[List[int]] = self._build_nonce_pool(
            pool_size=self.cfg.nonce_pool_size,
            nonce_len=self.cfg.nonce_tokens_len,
        )

        # 预编码 filler 片段（token ids）
        self.filler_ids_pool: List[List[int]] = [
            self.tok.encode(t, add_special_tokens=False) for t in self.cfg.filler_texts
        ]
        if any(len(x) == 0 for x in self.filler_ids_pool):
            raise ValueError("Some filler_texts encode to empty; please adjust filler_texts.")

    def build(self, target_tokens: int, seed: int = 0) -> str:
        """
        Build a prompt with EXACTLY `target_tokens` tokens (as counted by Qwen3 tokenizer),
        deterministic per (target_tokens, seed), and resistant to prefix caching.

        Note:
        - Uses add_special_tokens=False to keep token accounting stable.
        - If your serving stack auto-inserts BOS, set reserve_for_bos=1 in config.
        """
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive.")

        # 预留 BOS 空间（如果你的 serving 会自动加 BOS）
        effective_target = target_tokens - self.cfg.reserve_for_bos
        if effective_target <= 0:
            raise ValueError("target_tokens too small after reserve_for_bos.")

        # 选择一个固定长度的 nonce token 序列（由 seed 决定），注入到 prompt 早期
        nonce_ids = self.nonce_pool[seed % len(self.nonce_pool)]

        # 组装开始：prefix + nonce
        ids: List[int] = []
        ids.extend(self.prefix_ids)
        ids.extend(nonce_ids)

        if len(ids) > effective_target:
            # 如果 prefix+nonce 已经超过目标，直接截断（仍然精确）
            ids = ids[:effective_target]
            return self.tok.decode(ids, skip_special_tokens=False)

        # 填充到目标长度：用 deterministic 随机选择 filler 片段拼接
        rng = random.Random(seed)
        while len(ids) < effective_target:
            frag = self.filler_ids_pool[rng.randrange(len(self.filler_ids_pool))]
            need = effective_target - len(ids)
            if len(frag) <= need:
                ids.extend(frag)
            else:
                ids.extend(frag[:need])

        # 双保险：确保精确长度
        ids = ids[:effective_target]
        assert len(ids) == effective_target

        return self.tok.decode(ids, skip_special_tokens=False)

    def count_tokens(self, text: str) -> int:
        """Utility: count tokens with the same tokenizer settings."""
        return len(self.tok.encode(text, add_special_tokens=False))

    def _build_nonce_pool(self, pool_size: int, nonce_len: int) -> List[List[int]]:
        """
        Build a pool of nonce token-id sequences, each with exactly `nonce_len` tokens.
        We avoid variable tokenization length by constructing nonce directly in token-id space.
        """
        # 候选短字符串：尽量“tokenize 稳定”，包含空格/标记组合
        nonce_fragments = [
            " nonceA ",
            " nonceB ",
            " nonceC ",
            " nonceD ",
            " idXX ",
            " tagYY ",
            " seedZZ ",
            " keyQQ ",
            " alpha ",
            " beta ",
            " gamma ",
            " delta ",
        ]
        frag_ids = [self.tok.encode(s, add_special_tokens=False) for s in nonce_fragments]
        frag_ids = [x for x in frag_ids if len(x) > 0]
        if not frag_ids:
            raise ValueError("Failed to build nonce fragments; tokenizer produced empty encodings.")

        pool: List[List[int]] = []
        # 用不同的 RNG 种子构造 pool_size 个 nonce，每个严格 nonce_len
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
    cfg = PromptBuilderConfig(model_id="Qwen/Qwen3-8B", nonce_tokens_len=16, reserve_for_bos=0)
    builder = Qwen3PromptBuilder(cfg)

    for t in [1024, 2048, 4096, 8192]:
        p = builder.build(target_tokens=t, seed=1)
        print(t, builder.count_tokens(p))
