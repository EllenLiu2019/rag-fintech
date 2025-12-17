#!/usr/bin/env python3
"""
从 OMOP CDM Vocabulary 数据中提取 SNOMED 概念并构建包含中文 FSN 的数据集

用法:
    python extract_snomed.py
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

ENGLISH_LANGUAGE_ID = 4180186
CHINESE_LANGUAGE_ID = 4182948

# FSN 模式：匹配以小括号结尾的字符串，括号内可以包含空格等字符
# 例如: "Benefit application rejected (finding)"
#      "Product containing benomyl (medicinal product)"
FSN_PATTERNS = r"\([^)]+\)$"


def extract_fsn(row: pd.Series, synonym_df: pd.DataFrame) -> pd.Series:
    """
    从 CONCEPT_SYNONYM 中提取 FSN (Fully Specified Name)

    FSN 特征:
    1. 中文（language_concept_id = 4182948）
    2. 英文（language_concept_id = 4180186）包含语义标签（如 (finding), (disorder)）
    """
    fsn_name = synonym_df[
        (synonym_df["concept_id"] == row["concept_id"])
        & (synonym_df["language_concept_id"] == ENGLISH_LANGUAGE_ID)
        & (synonym_df["concept_synonym_name"].str.contains(FSN_PATTERNS, na=False, regex=True, case=False))
    ]["concept_synonym_name"].tolist()
    if len(fsn_name) > 0:
        row["FSN"] = "; ".join(fsn_name)
    else:
        row["FSN"] = ""
    return row


def build_vocabulary_dataset(
    concept_df: pd.DataFrame,
    synonym_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建包含 FSN 的词汇表数据集

    Args:
        concept_df: CONCEPT 表
        synonym_df: CONCEPT_SYNONYM 表
        vocabulary_id: 词汇表ID（默认 SNOMED）
    """
    # 使用 tqdm 显示处理进度
    tqdm.pandas(desc="Extracting FSN")
    result = concept_df.progress_apply(lambda row: extract_fsn(row, synonym_df), axis=1)

    print("\nFinal dataset statistics:")
    print(f"  Total concepts: {len(result):,}")
    print(f"  Concepts with FSN: {result['FSN'].notna().sum():,}")
    standard_count = (result["standard_concept"] == "S").sum()
    print(f"  Standard concepts: {standard_count:,}")
    valid_count = (result["valid_end_date"] == 20991231).sum()
    print(f"  Valid concepts: {valid_count:,}")

    return result


if __name__ == "__main__":
    sample_size = None  # 10000
    fsn = False

    vocabulary_dir = Path(__file__).parent.parent / "data" / "vocabulary_v5"
    concept_path = vocabulary_dir / "CONCEPT.csv"
    synonym_path = vocabulary_dir / "CONCEPT_SYNONYM.csv"

    print("=" * 60)
    print("SNOMED Vocabulary Extraction Tool")
    print("=" * 60)

    # Step 1: 读取 CONCEPT.csv 并筛选 SNOMED 数据
    print("\n[Step 1/4] Reading CONCEPT.csv and filtering SNOMED concepts...")
    snomed_concepts = []
    chunk_size = 1024

    # 估算总行数（用于进度条）
    try:
        total_lines = sum(1 for _ in open(concept_path, "r", encoding="utf-8"))
        total_chunks = (total_lines + chunk_size - 1) // chunk_size
    except Exception:
        total_chunks = None

    with tqdm(total=total_chunks, desc="Reading chunks", unit="chunk") as pbar:
        for chunk in pd.read_csv(concept_path, sep="\t", chunksize=chunk_size, low_memory=False):
            snomed_chunk = chunk[
                (chunk["vocabulary_id"] == "SNOMED")
                & (chunk["invalid_reason"].isna())
                & (chunk["valid_end_date"] == 20991231)
            ]
            if len(snomed_chunk) > 0:
                snomed_concepts.append(snomed_chunk)
            pbar.update(1)
            pbar.set_postfix({"SNOMED found": sum(len(c) for c in snomed_concepts)})

    concept_df = pd.concat(snomed_concepts, ignore_index=True)
    print(f"✓ Found {len(concept_df):,} SNOMED concepts")

    # Step 2: 采样（如果需要）
    if sample_size:
        if len(concept_df) > sample_size:
            print(f"\n[Step 2/4] Sampling {sample_size:,} concepts from {len(concept_df):,}...")
            concept_df = concept_df.sample(n=sample_size, random_state=42)
            print(f"✓ Sampled {len(concept_df):,} concepts")
        else:
            print(f"\n[Step 2/4] Skipping sampling (total concepts: {len(concept_df):,})")

    # Step 3: 读取 CONCEPT_SYNONYM.csv
    if fsn:
        print("\n[Step 3/4] Loading CONCEPT_SYNONYM.csv...")
        try:
            # 尝试获取文件大小用于进度估算
            file_size = synonym_path.stat().st_size
            synonym_df = pd.read_csv(str(synonym_path), sep="\t", low_memory=False, on_bad_lines="skip")
            print(f"✓ Loaded {len(synonym_df):,} synonym records")
        except Exception as e:
            print(f"✗ Error loading synonym file: {e}")
            raise

        # Step 4: 构建数据集并提取 FSN
        print("\n[Step 4/4] Building vocabulary dataset with FSN extraction...")
        result_df = build_vocabulary_dataset(concept_df, synonym_df)
        output_path = "tools/data/SNOMED_WITH_FSN.csv"
    else:
        result_df = concept_df
        output_path = "SNOMED_CONCEPT_ONLY.csv"

    print(f"\nSaving results to {output_path}...")
    result_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("✓ Extraction completed successfully!")
    print(f"  Output file: {output_path}")
    print(f"  Total concepts saved: {len(result_df):,}")
    print("=" * 60)
