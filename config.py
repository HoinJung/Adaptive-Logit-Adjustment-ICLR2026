"""
Configuration for data and classifier paths.
Override via environment variables:
  ALA_DATA_DIR, ALA_NLP_CLASSIFICATION_DIR, ALA_EMBEDDING_DIR
"""
import os

# Base directory for datasets (e.g. FACET annotations and images)
DATA_DIR = os.environ.get("ALA_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

# Directory containing trained text classifiers and token importance JSONs
NLP_CLASSIFICATION_DIR = os.environ.get(
    "ALA_NLP_CLASSIFICATION_DIR",
    os.path.join(os.path.dirname(__file__), "nlp_classification"),
)

# Directory for image/decoder embeddings (per-task embedding folders can live under tasks/*/embedding)
EMBEDDING_DIR = os.environ.get("ALA_EMBEDDING_DIR", os.path.join(os.path.dirname(__file__), "embedding"))

# FACET paths (relative to DATA_DIR)
FACET_ANNOTATIONS = os.path.join(DATA_DIR, "facet", "annotations", "annotations.csv")
FACET_NEW_ANNOTATIONS = os.path.join(DATA_DIR, "facet", "new_annotations.csv")
