"""finetuning_embeddings package.

Package for fine-tuning embedding models for NirmatAI.
"""

# Expose embedding_utils as a submodule only
import finetuning_embeddings.embedding_utils as embedding_utils
import finetuning_embeddings.utils as utils
from finetuning_embeddings.data_generator import DataGenerator
from finetuning_embeddings.embedding_data_processor import EmbeddingDataProcessor

__all__ = [
    "DataGenerator",
    "EmbeddingDataProcessor",
    "embedding_utils",
    "utils",
]
