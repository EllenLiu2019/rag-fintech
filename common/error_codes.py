# common/error_codes.py


class ErrorCodes:
    """
    Error Code Specification:
    - Format: {LAYER}{MODULE}{SEQ}
    - LAYER: A=API, S=Service, L=LLM, R=Repository
    - MODULE: 3-digit module code
    - SEQ: 3-digit sequence number
    """

    # API Layer (A)
    A_VALIDATION_001 = "A_VAL_001"  # Parameter validation failed
    A_AUTH_001 = "A_AUTH_001"  # Authentication failed
    A_NOTFOUND_001 = "A_NF_001"  # Resource not found
    A_RATELIMIT_001 = "A_RL_001"  # Frequency limit exceeded

    # Service Layer (S)
    S_INGESTION_001 = "S_ING_001"  # Document ingestion failed
    S_INGESTION_002 = "S_ING_002"  # Document parsing failed
    S_INGESTION_003 = "S_ING_003"  # Information extraction failed
    S_INGESTION_004 = "S_ING_004"  # Document chunking failed
    S_INGESTION_005 = "S_ING_005"  # Failed to enqueue task
    S_INGESTION_006 = "S_ING_006"  # Redis disabled
    S_INGESTION_007 = "S_ING_007"  # Failed to get task
    S_INGESTION_008 = "S_ING_008"  # Failed to update job progress
    S_RETRIEVAL_001 = "S_RET_001"  # Retrieval failed
    S_RETRIEVAL_002 = "S_RET_002"  # Reranking failed
    S_GENERATION_001 = "S_GEN_001"  # Generation failed

    # LLM Layer (L)
    L_MODEL_001 = "L_MOD_001"  # Model not found
    L_MODEL_002 = "L_MOD_002"  # Model timeout
    L_MODEL_003 = "L_MOD_003"  # Model rate limit exceeded
    L_EMBEDDING_001 = "L_EMB_001"  # Embedding failed
    L_TOKEN_001 = "L_TOK_001"  # Token limit exceeded

    # Repository Layer (R)
    R_VECTOR_001 = "R_VEC_001"  # Milvus connection failed
    R_VECTOR_002 = "R_VEC_002"  # Milvus query failed
    R_DB_001 = "R_DB_001"  # PostgreSQL connection failed
    R_DB_002 = "R_DB_002"  # PostgreSQL query failed
    R_DB_003 = "R_DB_003"  # Document not found
    R_CACHE_001 = "R_CACHE_001"  # Redis connection failed
    R_FILE_001 = "R_FILE_001"  # File storage failed
