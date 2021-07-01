from datetime import datetime
from typing import List, Dict

import geojson

from http_request.handler import HttpHandler
from milvus import MILVUS_DATABASE_HOST, MILVUS_DATABASE_PORT
from http_request.constants import MetricType, IndexType
from milvus.settings import BASE_DIR

http_handler = HttpHandler(host=MILVUS_DATABASE_HOST,
                           port=MILVUS_DATABASE_PORT)


def ping():
    response = http_handler.ping(timeout=10)
    print(f"Ping response: {response}\n")
    return response


def create_collection(collection_name: str, dimension: int,
                      index_file_size: int, metric_type: MetricType):
    response = http_handler.create_collection(collection_name=collection_name,
                                              dimension=dimension,
                                              index_file_size=index_file_size,
                                              metric_type=metric_type)
    print(f"Create collection response: {response}\n")


def show_collections(timeout: int):
    response = http_handler.show_collections(timeout=timeout)
    print(f"Show collection response: {response}\n")


def has_collection(collection_name: str, timeout: int):
    response = http_handler.has_collection(collection_name=collection_name,
                                           timeout=timeout)
    print(f"Has collection response: {response}\n")


def show_collection_info(collection_name: str, timeout: int):
    response = http_handler.show_collection_info(
        collection_name=collection_name, timeout=timeout)
    print(f"Show collection info response: {response}\n")


def drop_collection(collection_name: str, timeout: int):
    response = http_handler.drop_collection(collection_name=collection_name,
                                            timeout=timeout)
    print(f"Drop collection response: {response}\n")


def create_index(collection_name: str, index_type: IndexType,
                 index_params: Dict, timeout: int):
    response = http_handler.create_index(collection_name=collection_name,
                                         index_type=index_type,
                                         index_params=index_params,
                                         timeout=timeout)
    print(f"Create collection index response: {response}\n")


def describe_index(collection_name: str, timeout: int):
    response = http_handler.describe_index(collection_name=collection_name,
                                           timeout=timeout)
    print(f"Describe index response: {response}\n")


def create_partition(collection_name: str,
                     partition_tag: str,
                     timeout: int = 10):
    response = http_handler.create_partition(collection_name=collection_name,
                                             partition_tag=partition_tag,
                                             timeout=timeout)
    print(f"Create collection partition response: {response}\n")


def show_partition(collection_name: str,
                   timeout: int,
                   offset: int = 0,
                   page_size: int = 100):
    response = http_handler.show_partitions(collection_name=collection_name,
                                            offset=offset,
                                            page_size=page_size,
                                            timeout=timeout)
    print(f"Show collection partition response: {response}\n")


def _add_vector(collection_name: str,
                records,
                ids: List = None,
                partition_tag: str = None):
    return http_handler.add_vectors(collection_name=collection_name,
                                    records=records,
                                    partition_tag=partition_tag)


def add_vector(collection_name: str,
               file_path: str,
               ids: List = None,
               partition_tag: str = None):
    with open(file_path, mode='r', encoding='utf-8') as file:
        data_set = geojson.load(file)

    temp_sum = 0
    if data_set:
        start_t = datetime.now()
        features = data_set['features']
        for item in features:
            records = item['geometry']['coordinates']
            temp_sum += len(records[0])

            response = _add_vector(collection_name=collection_name,
                                   records=records[0],
                                   partition_tag=partition_tag)
            print(f"Create vector for: {response}\n")

        end_t = datetime.now()
        print(
            f"Total info: {temp_sum}, started at: {start_t} and ended at: {end_t}"
        )


def search_vectors(collection_name: str,
                   top_k: int,
                   query_records,
                   partition_tags: List = None,
                   search_params: Dict = None,
                   **kwargs):
    response = http_handler.search_vectors(collection_name=collection_name,
                                           top_k=top_k,
                                           query_records=query_records,
                                           partition_tags=partition_tags,
                                           search_params=search_params,
                                           kwargs=kwargs)
    print(f"Show collection partition response: {response}\n")


if __name__ == '__main__':
    if ping():
        # create_collection(collection_name='Monitoring_Trends',
        #                   dimension=2,
        #                   index_file_size=1024,
        #                   metric_type=MetricType.L2)
        #
        # show_collections(timeout=30)
        #
        # has_collection(collection_name='Monitoring_Trends', timeout=30)
        #
        # show_collection_info(collection_name='Monitoring_Trends', timeout=100)
        #
        # drop_collection(collection_name='Monitoring_Trends', timeout=30)
        #
        # create_index(collection_name='Monitoring_Trends',
        #              index_type=IndexType.IVFLAT,
        #              index_params={"nlist": 18384},
        #              timeout=30)
        #
        # create_partition(collection_name='Monitoring_Trends',
        #                  partition_tag='trend',
        #                  timeout=10)
        #
        # describe_index(collection_name='Monitoring_Trends', timeout=30)
        #
        # show_partition(collection_name='Monitoring_Trends', timeout=32)
        #
        # file_path = f"{BASE_DIR}/vector-dataset/Monitoring_Trends_in_Burn_Severity.geojson"
        # add_vector(collection_name='Monitoring_Trends',
        #            file_path=file_path,
        #            partition_tag='trend')

        search_vectors(
            collection_name='Monitoring_Trends',
            top_k=20,
            query_records=[[-152.267982254686956, 57.624012394958342]],
            partition_tags=['trend'],
            search_params={"nprobe": 16})
