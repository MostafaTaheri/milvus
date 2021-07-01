import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Milvus database information
MILVUS_DATABASE_HOST = os.environ.get('MILVUS_DATABASE_HOST', '192.168.18.24')
MILVUS_DATABASE_PORT = os.environ.get('MILVUS_DATABASE_PORT', 30111)
