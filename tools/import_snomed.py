import os
from pymilvus import DataType, MilvusClient
import pandas as pd
from tqdm import tqdm
import voyageai


def create_collection(client: MilvusClient, collection_name: str):
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
        description="SNOMED CT",
    )

    schema.add_field(field_name="id", is_primary=True, auto_id=True, datatype=DataType.INT64)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="concept_id", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="concept_name", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="domain_id", datatype=DataType.VARCHAR, max_length=20)
    schema.add_field(field_name="vocabulary_id", datatype=DataType.VARCHAR, max_length=20)
    schema.add_field(field_name="concept_class_id", datatype=DataType.VARCHAR, max_length=20)
    schema.add_field(field_name="standard_concept", datatype=DataType.VARCHAR, max_length=10)
    schema.add_field(field_name="concept_code", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="valid_start_date", datatype=DataType.VARCHAR, max_length=10)
    schema.add_field(field_name="valid_end_date", datatype=DataType.VARCHAR, max_length=10)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector", index_name="dense_vector_index", index_type="AUTOINDEX", metric_type="COSINE"
    )

    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )


def safe_str(value, default: str = "") -> str:
    """Safely convert value to string, handling NaN and None."""
    if pd.isna(value):
        return default
    return str(value).strip()


def prepare_row_data(row: pd.Series, embedding: list) -> dict:
    """Prepare a single row for insertion, returns None if invalid."""
    return {
        "concept_id": safe_str(row["concept_id"]),
        "concept_name": safe_str(row["concept_name"]),
        "domain_id": safe_str(row["domain_id"]),
        "vocabulary_id": safe_str(row["vocabulary_id"]),
        "concept_class_id": safe_str(row["concept_class_id"]),
        "standard_concept": safe_str(row["standard_concept"]),
        "concept_code": safe_str(row["concept_code"]),
        "valid_start_date": safe_str(row["valid_start_date"]),
        "valid_end_date": safe_str(row["valid_end_date"]),
        "dense_vector": embedding,
    }


def insert_snomed_data(
    voyage_client: voyageai.Client,
    milvus_client: MilvusClient,
    collection_name: str,
    batch_size: int,
    start_batch: int = 0,
):
    """
    Insert SNOMED data with graceful degradation.

    Args:
        voyage_client: VoyageAI client for embedding
        milvus_client: Milvus client for storage
        collection_name: Target collection name
        batch_size: Number of rows per batch
        start_batch: Skip first N batches (for resuming failed imports)
    """
    snomed_path = os.path.join(os.path.dirname(__file__), "data", "SNOMED_CONCEPT_ONLY.csv")

    # Statistics tracking
    stats = {
        "total_processed": 0,
        "total_inserted": 0,
        "embed_failures": 0,
        "insert_failures": 0,
        "skipped_rows": 0,
        "failed_batches": [],
        "failed_concept_ids": [],  # Track failed concept_ids for retry
    }

    # Use chunksize for efficient memory usage (O(n) instead of O(n²))
    chunk_iter = pd.read_csv(snomed_path, chunksize=batch_size)

    # Estimate total batches for progress bar
    try:
        total_lines = sum(1 for _ in open(snomed_path, "r", encoding="utf-8")) - 1  # -1 for header
        total_batches = (total_lines + batch_size - 1) // batch_size
    except Exception:
        total_batches = None

    with tqdm(total=total_batches, desc="Inserting SNOMED CT", unit="batch") as pbar:
        for batch_idx, chunk in enumerate(chunk_iter):
            # Skip batches if resuming
            if batch_idx < start_batch:
                pbar.update(1)
                pbar.set_postfix({"status": f"skipping batch {batch_idx}"})
                continue

            stats["total_processed"] += len(chunk)
            batch_start = batch_idx * batch_size
            batch_end = batch_start + len(chunk)

            # Step 1: Prepare texts for embedding
            texts_to_embed = []
            valid_indices = []  # Track which rows have valid concept_name

            for idx, (_, row) in enumerate(chunk.iterrows()):
                concept_name = safe_str(row.get("concept_name", ""))
                concept_id = safe_str(row.get("concept_id", ""))
                if concept_name:
                    texts_to_embed.append(concept_name)
                    valid_indices.append(idx)
                else:
                    stats["skipped_rows"] += 1
                    if concept_id:
                        stats["failed_concept_ids"].append(
                            {
                                "concept_id": concept_id,
                                "reason": "empty_concept_name",
                            }
                        )

            if not texts_to_embed:
                pbar.update(1)
                continue

            # Step 2: Generate embeddings
            # Batch-level failure -> only record in failed_batches (not individual concept_ids)
            embeddings = None
            try:
                embeddings = voyage_client.embed(
                    texts=texts_to_embed, model="voyage-3.5-lite", input_type="document"
                ).embeddings
            except Exception as e:
                stats["embed_failures"] += len(texts_to_embed)
                stats["failed_batches"].append(
                    {
                        "batch": batch_idx,
                        "count": len(texts_to_embed),
                        "error": f"embed: {e}",
                    }
                )
                print(f"\n⚠ Batch {batch_idx} ({batch_start}-{batch_end}): Embedding failed - {e}")
                pbar.update(1)
                continue

            # Step 3: Prepare data with per-row error handling
            data_to_insert = []
            for i, valid_idx in enumerate(valid_indices):
                row = chunk.iloc[valid_idx]
                concept_id = safe_str(row.get("concept_id", ""))
                try:
                    row_data = prepare_row_data(row, embeddings[i])
                    data_to_insert.append(row_data)
                except Exception as e:
                    stats["skipped_rows"] += 1
                    stats["failed_concept_ids"].append(
                        {
                            "concept_id": concept_id,
                            "reason": f"prepare_data: {str(e)[:100]}",
                        }
                    )

            if not data_to_insert:
                pbar.update(1)
                continue

            # Step 4: Insert to Milvus with error handling
            try:
                milvus_client.insert(collection_name=collection_name, data=data_to_insert)
                stats["total_inserted"] += len(data_to_insert)
            except Exception as e:
                stats["insert_failures"] += len(data_to_insert)
                # Batch-level failure -> only record in failed_batches (not individual concept_ids)
                stats["failed_batches"].append(
                    {
                        "batch": batch_idx,
                        "count": len(data_to_insert),
                        "error": f"insert: {e}",
                    }
                )
                print(f"\n⚠ Batch {batch_idx} ({batch_start}-{batch_end}): Insert failed - {e}")

            pbar.update(1)
            pbar.set_postfix(
                {
                    "inserted": stats["total_inserted"],
                    "failed": stats["embed_failures"] + stats["insert_failures"],
                }
            )

    # Print summary
    print("\n" + "=" * 60)
    print("Import Summary")
    print("=" * 60)
    print(f"  Total rows processed: {stats['total_processed']:,}")
    print(f"  Successfully inserted: {stats['total_inserted']:,}")
    print(f"  Embedding failures: {stats['embed_failures']:,}")
    print(f"  Insert failures: {stats['insert_failures']:,}")
    print(f"  Skipped rows (invalid): {stats['skipped_rows']:,}")
    print(f"  Failed concept_ids: {len(stats['failed_concept_ids']):,}")

    # Show failed batches (batch-level failures)
    if stats["failed_batches"]:
        total_batch_failures = sum(fb.get("count", 0) for fb in stats["failed_batches"])
        print(f"\n  Failed batches: {len(stats['failed_batches'])} batches ({total_batch_failures:,} rows)")
        for fb in stats["failed_batches"][:10]:  # Show first 10
            print(f"    - Batch {fb['batch']} ({fb.get('count', '?')} rows): {fb['error']}")
        if len(stats["failed_batches"]) > 10:
            print(f"    ... and {len(stats['failed_batches']) - 10} more batches")

    # Show failed concept_ids (row-level failures)
    if stats["failed_concept_ids"]:
        failed_file = os.path.join(os.path.dirname(__file__), "data", "failed_concept_ids.csv")
        failed_df = pd.DataFrame(stats["failed_concept_ids"])
        failed_df.to_csv(failed_file, index=False)
        print(f"\n  Failed rows: {len(stats['failed_concept_ids']):,} concept_ids")
        print(f"    → Saved to: {failed_file}")
        print("    Sample:")
        for fc in stats["failed_concept_ids"][:5]:
            print(f"      - {fc['concept_id']}: {fc['reason']}")
        if len(stats["failed_concept_ids"]) > 5:
            print(f"      ... and {len(stats['failed_concept_ids']) - 5} more")

    print("=" * 60)
    return stats


if __name__ == "__main__":
    # Configuration
    IMPORT_SNOMED = True
    BATCH_SIZE = 100  # Voyage API has rate limits, smaller batches are safer
    START_BATCH = 0  # Set > 0 to resume from a specific batch after failure

    # Initialize clients
    milvus_client = MilvusClient(
        uri="https://in03-f663856b627ed8e.serverless.aws-eu-central-1.cloud.zilliz.com",
        token=os.getenv("RAG_MILVUS_TOKEN"),
    )
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    collection_name = "rag_fintech_snomed_kb"

    # Create collection if needed
    if not milvus_client.has_collection(collection_name):
        print(f"Creating collection '{collection_name}'...")
        create_collection(milvus_client, collection_name)
        print("✓ Collection created")

    # Import data
    if IMPORT_SNOMED:
        stats = insert_snomed_data(
            voyage_client=voyage_client,
            milvus_client=milvus_client,
            collection_name=collection_name,
            batch_size=BATCH_SIZE,
            start_batch=START_BATCH,
        )
