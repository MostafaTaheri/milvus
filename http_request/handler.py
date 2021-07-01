import json
import logging
import struct
import requests
from typing import List, Dict

from .abstracts import MilvusAbstract, IndexParam, CollectionSchema
from .abstracts import TopKQueryResult, PartitionParam
from .constants import Status, IndexType, MetricType
from milvus import NotConnectError
from .handler_wrapper import handle_error

logger = logging.getLogger(__name__)

IndexValueNameMap = {
    IndexType.INVALID: "INVALID",
    IndexType.FLAT: "FLAT",
    IndexType.IVFLAT: "IVFFLAT",
    IndexType.IVF_SQ8: "IVFSQ8",
    IndexType.IVF_SQ8H: "IVFSQ8H",
    IndexType.IVF_PQ: "IVFPQ",
    IndexType.RNSG: "RNSG",
    IndexType.HNSW: "HNSW",
    IndexType.ANNOY: "ANNOY"
}

IndexNameValueMap = {
    "INVALID": IndexType.INVALID,
    "FLAT": IndexType.FLAT,
    "IVFFLAT": IndexType.IVFLAT,
    "IVFSQ8": IndexType.IVF_SQ8,
    "IVFSQ8H": IndexType.IVF_SQ8H,
    "IVFPQ": IndexType.IVF_PQ,
    "RNSG": IndexType.RNSG,
    "HNSW": IndexType.HNSW,
    "ANNOY": IndexType.ANNOY
}

MetricValueNameMap = {
    MetricType.L2: "L2",
    MetricType.IP: "IP",
    MetricType.HAMMING: "HAMMING",
    MetricType.JACCARD: "JACCARD",
    MetricType.TANIMOTO: "TANIMOTO",
    MetricType.SUBSTRUCTURE: "SUBSTRUCTURE",
    MetricType.SUPERSTRUCTURE: "SUPERSTRUCTURE"
}


class HttpHandler(MilvusAbstract):
    """
    Client http handler class
    """
    def __init__(self, host: str, port: int, **kwargs):
        self._status = None
        self._uri = self._set_uri(host=host, port=port)
        self._max_retry = kwargs.get("max_retry", 3)

    def __enter__(self):
        self.ping()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def _set_uri(host: str, port: int):
        """
        Set server network address
        """
        return "http://{}:{}".format(host, port)

    @property
    def status(self):
        """
        Show the connection status
        """
        return self._status

    def ping(self, timeout: int = 10):
        """
        Check the network connectivity
        """
        logging.info("Connecting server {}".format(self._uri))
        retry = self._max_retry
        try:
            while retry > 0:
                try:
                    requests.get(self._uri + "/state", timeout=timeout)
                    return True
                except:
                    retry -= 1
                    if retry > 0:
                        continue
                    else:
                        raise
        except Exception as ex:
            logger.error("Cannot connect server {}... {}".format(
                self._uri, str(ex)))
            raise NotConnectError("Cannot get server status")

        logger.info("Connected server {}".format(self._uri))

    def _set_config(self, cmd, timeout: int):
        """
        Set configuration for Milvus database
        """
        if cmd.startswith("set_config"):
            cmd_node = cmd.split(" ")
            config_node = cmd_node[1].split(".")
            request = {config_node[0]: {config_node[1]: cmd_node[2]}}

            url = self._uri + "/system/config"
            payload = json.dumps(request)
            response = requests.put(url, data=payload, timeout=timeout)
            if response.status_code == 200:
                js = response.json()
                return Status(), js["message"]
            elif response.status_code == 400:
                js = response.json()
                return Status(js["code"], js["message"]), None
            else:
                return Status(Status.UNEXPECTED_ERROR, response.reason)

    def _get_config(self, cmd, timeout: int):
        """
        Show milvus database configuration information.
        """
        if cmd.startswith("get_config"):
            cmd_node = cmd.split(" ")
            config_node = cmd_node[1].split(".")

            url = self._uri + "/system/config"
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                js = response.json()
                rc_parent = js.get(config_node[0], None)
                if rc_parent is None:
                    return Status(
                        Status.UNEXPECTED_ERROR,
                        "Config {} not supported".format(cmd_node[1]))
                rc_child = rc_parent.get(config_node[1], None)
                if rc_child is None:
                    return Status(
                        Status.UNEXPECTED_ERROR,
                        "Config {} not supported".format(cmd_node[1]))

                return Status(), rc_child
            elif response.status_code == 400:
                js = response.json()
                return Status(js["code"], js["message"]), None
            else:
                return Status(Status.UNEXPECTED_ERROR, response.reason)

    @handle_error(returns=(None, ))
    def _cmd(self, cmd, timeout=10):
        if cmd.startswith("get_config"):
            return self._get_config(cmd, timeout)
        if cmd.startswith("set_config"):
            return self._set_config(cmd, timeout)

        url = self._uri + "/system/{}".format(cmd)
        response = requests.get(url, timeout=timeout)
        js = response.json()
        if response.status_code == 200:
            return Status(), js["reply"]

        return Status(code=js["code"], message=js["message"]), None

    def server_version(self, timeout: int):
        """
        Show the version of server
        """
        return self._cmd("version", timeout)

    def server_status(self, timeout):
        """
        Show the version of server
        """
        return self._cmd("status", timeout)

    @handle_error()
    def create_collection(self, collection_name: str, dimension: int,
                          index_file_size: int, metric_type: MetricType):
        """
        Create collection

        :type  collection_name: str
        :param collection_name: the name of collection

        :type  dimension: int
        :param dimension: the size of collection dimension

        :type  index_file_size: int
        :param index_file_size: specify the size of collection

        :type  metric_type: MetricType
        :param metric_type:

        :return: Status, indicate if connect is successful
        """
        metric = MetricValueNameMap.get(metric_type, None)
        table_param = {
            "collection_name": collection_name,
            "dimension": dimension,
            "index_file_size": index_file_size,
            "metric_type": metric
        }
        data = json.dumps(table_param)
        url = self._uri + "/collections"
        try:
            response = requests.post(url, data=data)
            if response.status_code == 201:
                return Status(message='Create table successfully!')

            js = response.json()
            return Status(js["code"], js["message"])
        except Exception as ex:
            return Status(Status.UNEXPECTED_ERROR, message=str(ex))

    @handle_error(returns=(False, ))
    def has_collection(self, collection_name: str, timeout: int):
        """

        This method is used to test table existence.

        :type collection_name: str
        :param collection_name: collection name is going to be tested.

        :type  timeout: int
        :param timeout:

        :return:
            has_table: bool, if given table_name exists

        """
        url = self._uri + "/collections/" + collection_name
        try:
            response = requests.get(url=url, timeout=timeout)
            if response.status_code == 200:
                return Status(), True

            if response.status_code == 404:
                return Status(), False

            js = response.json()
            return Status(js["code"], js["message"]), False
        except Exception as ex:
            return Status(Status.UNEXPECTED_ERROR, message=str(ex))

    @handle_error(returns=(None, ))
    def get_table_row_count(self, table_name: str, timeout: int):
        """
        Get table row count

        :type  table_name, str
        :param table_name, target table name.

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :returns:
            Status: indicate if operation is successful
            count: int, table row count
        """
        url = self._uri + "/collections/{}".format(table_name)
        try:
            response = requests.get(url, timeout=timeout)
            js = response.json()
            if response.status_code == 200:
                return Status(), js["count"]

            return Status(js["code"], js["message"]), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    @handle_error(returns=(None, ))
    def describe_collection(self, collection_name: str, timeout: int):
        """
        Show table information

        :type  collection_name: str
        :param collection_name: which table to be shown

        :type  timeout: int
        :param timeout:

        :returns:
            Status: indicate if query is successful
            table_schema: TableSchema, given when operation is successful
        """
        url = self._uri + "/collections/{}".format(collection_name)
        response = requests.get(url, timeout=timeout)
        if response.status_code >= 500:
            return Status(Status.UNEXPECTED_ERROR, response.reason), None

        js = response.json()
        if response.status_code == 200:
            metric_map = dict()
            temp = [
                metric_map.update({i.name: i.value}) for i in MetricType
                if i.value > 0
            ]
            table = CollectionSchema(collection_name=js["collection_name"],
                                     dimension=js["dimension"],
                                     index_file_size=js["index_file_size"],
                                     metric_type=metric_map[js["metric_type"]])
            return Status(message='Described table successfully!'), table

        return Status(js["code"], js["message"]), None

    @handle_error(returns=([], ))
    def show_collections(self, timeout: int):
        """
        Show all tables in database

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if this operation is successful
            tables: list[str], list of table names
        """
        url = self._uri + "/collections"
        response = requests.get(url,
                                params={
                                    "offset": 0,
                                    "page_size": 0
                                },
                                timeout=timeout)
        if response.status_code != 200:
            return Status(Status.UNEXPECTED_ERROR, response.reason), []

        js = response.json()
        count = js["count"]
        response = requests.get(url,
                                params={
                                    "offset": 0,
                                    "page_size": count
                                },
                                timeout=timeout)
        if response.status_code != 200:
            return Status(Status.UNEXPECTED_ERROR, response.reason), []

        tables = []
        js = response.json()
        for table in js["collections"]:
            tables.append(table["collection_name"])
        return Status(), tables

    @handle_error(returns=(None, ))
    def show_collection_info(self, collection_name: str, timeout: int = 10):
        """
        Show information of table state

        :type  collection_name: str
        :param collection_name: which table to be shown

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if this operation is successful
            query_results: information of state
        """
        url = self._uri + "/collections/{}?info=stat".format(collection_name)
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return Status(), response.json()

        if response.status_code == 404:
            return Status(Status.COLLECTION_NOT_EXISTS,
                          "Collection not found"), None

        if response.text:
            result = response.json()
            return Status(result["code"], result["message"]), None

        return Status(Status.UNEXPECTED_ERROR, "Response is empty"), None

    @handle_error()
    def preload_collection(self,
                           collection_name: str,
                           timeout: int,
                           partition_tags: List = None):
        """
        load table to memory cache in advance

        :param collection_name: target table name.
        :type collection_name: str

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :type  partition_tags: List
        :param partition_tags:

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """
        url = self._uri + "/system/task"
        params = {"load": {"collection_name": collection_name}}
        if partition_tags:
            params["load"]["partition_tags"] = partition_tags

        data = json.dumps(params)
        response = requests.put(url, data=data, timeout=timeout)
        if response.status_code == 200:
            return Status(message="Load successfully")

        js = response.json()
        return Status(code=js["code"], message=js["message"])

    @handle_error()
    def drop_collection(self, collection_name: str, timeout: int):
        """
        Drop collection

        :type  collection_name: str
        :param collection_name: collection name of the deleting table

        :type  timeout: int
        :param timeout:

        :return: Status, indicate if connect is successful
        """
        url = self._uri + "/collections/" + collection_name
        response = requests.delete(url, timeout=timeout)
        if response.status_code == 204:
            return Status(message="Delete successfully!")

        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error(returns=([], ))
    def add_vectors(self,
                    collection_name: str,
                    records,
                    ids: List = None,
                    partition_tag: str = None):
        """
        Add vectors to table

        :type  collection_name: str
        :param collection_name: collection name been inserted

        :type  records: list[RowRecord]
        :param records: list of vectors been inserted

        :type  ids: list[int]
        :param ids: list of ids

        :type  partition_tag: str
        :param partition_tag:

        :returns:
            Status : indicate if vectors inserted successfully
            ids :list of id, after inserted every vector is given a id
        """
        url = self._uri + "/collections/{}/vectors".format(collection_name)
        data_dict = dict()
        if ids:
            data_dict["ids"] = list(map(str, ids))

        if partition_tag:
            data_dict["partition_tag"] = partition_tag

        if isinstance(records[0], bytes):
            vectors = [struct.unpack(str(len(r)) + 'B', r) for r in records]
            data_dict["vectors"] = vectors
        else:
            data_dict["vectors"] = records

        data = json.dumps(data_dict)
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=data, headers=headers)
        js = response.json()
        if response.status_code == 201:
            ids = [int(item) for item in list(js["ids"])]
            return Status(message='Add vectors successfully!'), ids

        return Status(js["code"], js["message"]), []

    @handle_error(returns=(None, ))
    def get_vectors_by_ids(self, collection_name: str, ids: List,
                           timeout: int):
        status, table_schema = self.describe_collection(
            collection_name, timeout)
        if not status.ok():
            return status, None

        metric = table_schema.metric_type
        bin_vector = metric in list(MetricType.__members__.values())[3:]
        url = self._uri + "/collections/{}/vectors".format(collection_name)
        ids_list = list(map(str, ids))
        query_ids = ",".join(ids_list)
        url = url + "?ids=" + query_ids
        response = requests.get(url, timeout=timeout)
        result = response.json()

        if response.status_code == 200:
            vectors = result["vectors"]
            if not list(vectors):
                return Status(), []

            vector_results = []
            for vector_res in vectors:
                vector = list(vector_res["vector"])
                if bin_vector:
                    vector_results.append(bytes(vector))
                else:
                    vector_results.append(vector)
            return Status(), vector_results

        return Status(result["code"], result["message"]), None

    @handle_error(returns=(None, ))
    def get_vector_ids(self, collection_name: str, segment_name: str,
                       timeout: int):
        url = self._uri + "/collections/{}/segments/{}/ids?page_size=1000000".format(
            collection_name, segment_name)
        response = requests.get(url, timeout=timeout)
        result = response.json()

        if response.status_code == 200:
            return Status(), list(map(int, result["ids"]))

        return Status(result["code"], result["message"]), None

    @handle_error()
    def create_index(self, collection_name: str, index_type: IndexType,
                     index_params: Dict, timeout: int):
        """
        Create specified index in a table

        :type  collection_name: str
        :param collection_name: collection name

        :type index_type: IndexType
        :param index_type: index information dict

            example: index_type = IndexType.FLAT

        :type index_params: Dict
        :param index_params:

            example: index_params = {"nlist": 18384}

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if this operation is successful

        :rtype: Status
        """
        url = self._uri + "/collections/{}/indexes".format(collection_name)
        index = IndexValueNameMap.get(index_type)
        request = dict()
        request["index_type"] = index
        request["params"] = index_params
        data = json.dumps(request)
        headers = {"Content-Type": "application/json"}
        response = requests.post(url,
                                 data=data,
                                 headers=headers,
                                 timeout=timeout)
        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error(returns=(None, ))
    def describe_index(self, collection_name: str, timeout: int):
        """
        Show index information

        :param collection_name: target collection name.
        :type collection_name: str

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

            TableSchema: table detail information

        :rtype: (Status, TableSchema)
        """
        url = self._uri + "/collections/{}/indexes".format(collection_name)
        response = requests.get(url, timeout=timeout)
        if response.status_code >= 500:
            return Status(
                Status.UNEXPECTED_ERROR,
                "Unexpected error.\n\tStatus code : {}, reason : {}".format(
                    response.status_code, response.reason))

        js = response.json()
        if response.status_code == 200:
            index_type = IndexNameValueMap.get(js["index_type"])
            return Status(), IndexParam(collection_name, index_type,
                                        js["params"])

        return Status(js["code"], js["message"]), None

    @handle_error()
    def drop_index(self, collection_name: str, timeout: int):
        """
        Drop index

        :param collection_name: target collection name.
        :type collection_name: str

        :type  timeout: int
        :param timeout:

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """
        url = self._uri + "/collections/{}/indexes".format(collection_name)
        response = requests.delete(url)
        if response.status_code == 204:
            return Status()

        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error()
    def create_partition(self,
                         collection_name: str,
                         partition_tag: str,
                         timeout: int = 10):
        """
        Create partition

        :param collection_name: target collection name.
        :type collection_name: str

        :type  partition_tag:
        :param partition_tag:

        :type  timeout: int
        :param timeout:

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """
        url = self._uri + "/collections/{}/partitions".format(collection_name)
        data = json.dumps({"partition_tag": partition_tag})
        headers = {"Content-Type": "application/json"}
        response = requests.post(url,
                                 data=data,
                                 headers=headers,
                                 timeout=timeout)
        if response.status_code == 201:
            return Status()

        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error(returns=([], ))
    def show_partitions(self,
                        collection_name: str,
                        timeout: int,
                        offset: int = 0,
                        page_size: int = 100):
        """
        Show all partition of specific table

        :param collection_name: target table name.
        :type collection_name: str

        :type  offset: int
        :param offset:

        :type  page_size: int
        :param page_size:

        :type  timeout: int
        :param timeout:

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """
        url = self._uri + "/collections/{}/partitions".format(collection_name)
        query_data = {"offset": offset, "page_size": page_size}
        response = requests.get(url, params=query_data, timeout=timeout)
        if response.status_code >= 500:
            return Status(
                Status.UNEXPECTED_ERROR,
                "Unexpected error. Status code : 500, reason: {}".format(
                    response.reason)), None

        js = response.json()
        if response.status_code == 200:
            partition_list = [
                PartitionParam(collection_name, item["partition_tag"])
                for item in js["partitions"]
            ]
            return Status(), partition_list

        return Status(js["code"], js["message"]), []

    @handle_error(returns=(False, ))
    def has_partition(self, collection_name: str, tag: str, timeout: int = 30):
        """
        Check the table has partition or not

        :param collection_name: target collection name.
        :type collection_name: str

        :type  tag: str
        :param tag:

        :type  timeout: int
        :param timeout:

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """
        url = self._uri + "/collections/{}/partitions".format(collection_name)
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            if result["count"] > 0:
                partitions = [
                    p["partition_tag"] for p in list(result["partitions"])
                ]
                return Status(), tag in partitions
            return Status(), False

        js = response.json()
        return Status(js["code"], js["message"]), False

    @handle_error()
    def drop_partition(self,
                       collection_name: str,
                       partition_tag: str,
                       timeout: int = 10):
        """
        Drop a table partition

        :param collection_name: target collection name.
        :type collection_name: str

        :type  partition_tag: str
        :param partition_tag:

        :type  timeout: int
        :param timeout:

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """
        url = self._uri + "/collections/{}/partitions".format(collection_name)
        request = {"partition_tag": partition_tag}
        payload = json.dumps(request)
        response = requests.delete(url, data=payload, timeout=timeout)
        if response.status_code == 204:
            return Status()

        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error(returns=(None, ))
    def search_vectors(self,
                       collection_name: str,
                       top_k: int,
                       query_records,
                       partition_tags: List = None,
                       search_params: Dict = None,
                       **kwargs):
        """
        Query vectors in a table

        :type  collection_name: str
        :param collection_name: collection name name been queried

        :type  query_records: list[RowRecord]
        :param query_records: all vectors going to be queried

        :type  partition_tags: list
        :param partition_tags:

        :type  search_params: dict
        :param search_params:

            example: {"nprobe": 16}

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        url = self._uri + "/collections/{}/vectors".format(collection_name)
        search_body = dict()
        if partition_tags:
            search_body["partition_tags"] = partition_tags
        search_body["topk"] = top_k
        search_body["params"] = search_params
        if isinstance(query_records[0], bytes):
            vectors = [
                struct.unpack(str(len(r)) + 'B', r) for r in query_records
            ]
            search_body["vectors"] = vectors
        else:
            vectors = query_records
            search_body["vectors"] = vectors

        data = json.dumps({"search": search_body})
        headers = {"Content-Type": "application/json"}
        response = requests.put(url, data, headers=headers)

        if response.status_code == 200:
            return Status(), TopKQueryResult(response)

        js = response.json()
        return Status(js["code"], js["message"]), None

    @handle_error(returns=(None, ))
    def search_by_ids(self,
                      collection_name: str,
                      ids: List,
                      top_k: int,
                      partition_tags: List = None,
                      search_params: Dict = None,
                      timeout=None,
                      **kwargs):
        """
        Query vectors in a table by id

        :type  collection_name: str
        :param collection_name: collection name name been queried

        :type  ids: list
        :param ids: all vectors going to be queried

        :type  partition_tags: list
        :param partition_tags:

        :type  search_params: dict
        :param search_params:

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :type  timeout: int
        :param timeout:

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        url = self._uri + "/collections/{}/vectors".format(collection_name)
        body_dict = dict()
        body_dict["topk"] = top_k
        body_dict["ids"] = list(map(str, ids))
        if partition_tags:
            body_dict["partition_tags"] = partition_tags
        if search_params:
            body_dict["params"] = search_params

        data = json.dumps({"search": body_dict})
        headers = {"Content-Type": "application/json"}
        response = requests.put(url, data, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return Status(), TopKQueryResult(response)

        js = response.json()
        return Status(js["code"], js["message"]), None

    @handle_error(returns=(None, ))
    def search_vectors_in_files(self, collection_name: str, file_ids: List,
                                query_records: List, top_k: int,
                                search_params: Dict, timeout: int, **kwargs):
        """
        Query vectors in a table, query vector in specified files

        :type  collection_name: str
        :param collection_name: collection name been queried

        :type  file_ids: list[str]
        :param file_ids: Specified files id array

        :type  query_records: list[RowRecord]
        :param query_records: all vectors going to be queried

        :type  search_params: list
        :param search_params:

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :type  timeout: int
        :param timeout:

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        url = self._uri + "/collections/{}/vectors".format(collection_name)
        body_dict = dict()
        body_dict["topk"] = top_k
        body_dict["file_ids"] = list(map(str, file_ids))
        body_dict["params"] = search_params
        if isinstance(query_records[0], bytes):
            vectors = [
                struct.unpack(str(len(r)) + 'B', r) for r in query_records
            ]
            body_dict["vectors"] = vectors
        else:
            vectors = query_records
            body_dict["vectors"] = vectors

        data = json.dumps({"search": body_dict})
        headers = {"Content-Type": "application/json"}
        response = requests.put(url, data, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return Status(), TopKQueryResult(response)

        js = response.json()
        return Status(js["code"], js["message"]), None

    @handle_error()
    def delete_by_id(self,
                     collection_name: str,
                     id_array: List,
                     timeout: int = None):
        """
        Drop a table by id

        :param collection_name: target collection name.
        :type collection_name: str

        :type  id_array: list
        :param id_array:

        :type  timeout: int
        :param timeout:

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """
        url = self._uri + "/collections/{}/vectors".format(collection_name)
        headers = {"Content-Type": "application/json"}
        ids = list(map(str, id_array))
        request = {"delete": {"ids": ids}}
        response = requests.put(url,
                                data=json.dumps(request),
                                headers=headers,
                                timeout=timeout)
        result = response.json()
        return Status(result["code"], result["message"])

    @handle_error()
    def flush(self, collection_name_array: List):
        url = self._uri + "/system/task"
        headers = {"Content-Type": "application/json"}
        request = {"flush": {"collection_names": collection_name_array}}
        response = requests.put(url, json.dumps(request), headers=headers)
        result = response.json()
        return Status(result["code"], result["message"])

    @handle_error()
    def compact(self, collection_name):
        url = self._uri + "/system/task"
        headers = {"Content-Type": "application/json"}
        request = {"compact": {"collection_name": collection_name}}
        response = requests.put(url, json.dumps(request), headers=headers)
        result = response.json()
        return Status(result["code"], result["message"])
