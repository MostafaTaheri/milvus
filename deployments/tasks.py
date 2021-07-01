from datetime import datetime
from pathlib import Path

import geojson

from http_request.handler import HttpHandler
from milvus import MILVUS_DATABASE_HOST, MILVUS_DATABASE_PORT
from http_request.constants import MetricType, IndexType
from milvus.celery_config import app

http_handler = HttpHandler(host=MILVUS_DATABASE_HOST,
                           port=MILVUS_DATABASE_PORT)

BASE_DIR = Path(__file__).resolve().parent.parent


@app.task(name='deployments.tasks.add_vector', queue='milvus_worker')
def add_vector(symbol: str):
    file_path = f"{BASE_DIR}/vector-dataset/Monitoring_Trends_in_Burn_Severity.geojson"
    with open(file_path, mode='r', encoding='utf-8') as file:
        data_set = geojson.load(file)

    temp_sum = 0
    if data_set:
        start_t = datetime.now()
        features = data_set['features']
        for item in features:
            records = item['geometry']['coordinates']
            temp_sum += len(records[0])
            response = http_handler.add_vectors(
                collection_name='Monitoring_Trends',
                records=records[0],
                partition_tag='trend')

            print(
                f"Create vector for: {len(records[0])} vectors for {symbol}\n")

        end_t = datetime.now()
        print(
            f"Total info: {temp_sum} for symbol: {symbol}, started at: {start_t} and ended at: {end_t}"
        )
