from typing import List, Dict


class MilvusAbstract:
    """Client abstract class

    Connection is a abstract class
    """
    def create_collection(self, collection_name: str, dimension: int,
                          index_file_size: int, metric_type):
        """
        Create collection
        Should be implemented

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
        raise NotImplementedError('You need to override this function')

    def has_collection(self, collection_name: str, timeout: int):
        """

        This method is used to test table existence.
        Should be implemented

        :type collection_name: str
        :param collection_name: collection name is going to be tested.

        :type  timeout: int
        :param timeout:

        :return:
            has_table: bool, if given table_name exists

        """
        raise NotImplementedError('You need to override this function')

    def drop_collection(self, collection_name: str, timeout: int):
        """
        Drop collection
        Should be implemented

        :type  collection_name: str
        :param collection_name: collection name of the deleting table

        :type  timeout: int
        :param timeout:

        :return: Status, indicate if connect is successful
        """
        raise NotImplementedError('You need to override this function')

    def add_vectors(self, collection_name: str, records: List, ids: List = None, **kwargs):
        """
        Add vectors to table
        Should be implemented

        :type  collection_name: str
        :param collection_name: collection name been inserted

        :type  records: list[RowRecord]
        :param records: list of vectors been inserted

        :type  ids: list[int]
        :param ids: list of ids

        :returns:
            Status : indicate if vectors inserted successfully
            ids :list of id, after inserted every vector is given a id
        """
        raise NotImplementedError('You need to override this function')

    def search_vectors(self, table_name: str, top_k: int, query_records: List,
                       query_ranges: List, **kwargs):
        """
        Query vectors in a table
        Should be implemented

        :type  table_name: str
        :param table_name: table name been queried

        :type  query_records: list[RowRecord]
        :param query_records: all vectors going to be queried

        :type  query_ranges: list[Range]
        :param query_ranges: Optional ranges for conditional search.
            If not specified, search whole table

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        raise NotImplementedError('You need to override this function')

    def search_vectors_in_files(self, table_name: str, file_ids: List,
                                query_records: List, top_k: int,
                                query_ranges: List, **kwargs):
        """
        Query vectors in a table, query vector in specified files
        Should be implemented

        :type  table_name: str
        :param table_name: table name been queried

        :type  file_ids: list[str]
        :param file_ids: Specified files id array

        :type  query_records: list[RowRecord]
        :param query_records: all vectors going to be queried

        :type  query_ranges: list[Range]
        :param query_ranges: Optional ranges for conditional search.
            If not specified, search whole table

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        raise NotImplementedError('You need to override this function')

    def describe_collection(self, collection_name: str, timeout: int):
        """
        Show table information
        Should be implemented

        :type  collection_name: str
        :param collection_name: which table to be shown

        :type  timeout: int
        :param timeout:

        :returns:
            Status: indicate if query is successful
            table_schema: TableSchema, given when operation is successful
        """
        raise NotImplementedError('You need to override this function')

    def get_table_row_count(self, table_name: str, timeout: int):
        """
        Get table row count
        Should be implemented

        :type  table_name, str
        :param table_name, target table name.

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :returns:
            Status: indicate if operation is successful
            count: int, table row count
        """
        raise NotImplementedError('You need to override this function')

    def show_collections(self, timeout: int):
        """
        Show all tables in database
        should be implemented

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if this operation is successful
            tables: list[str], list of table names
        """
        raise NotImplementedError('You need to override this function')

    def create_index(self, table_name: str, index_type, index_params: Dict, timeout: int):
        """
        Create specified index in a table
        should be implemented

        :type  table_name: str
        :param table_name: table name

         :type index_type: IndexType
        :param index_type: index information dict

            example: index_type = IndexType.FLAT

        :type index_params: Dict
        :param index_params:

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if this operation is successful

        :rtype: Status
        """
        raise NotImplementedError('You need to override this function')

    def server_version(self, timeout: int):
        """
        Provide server version
        should be implemented

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        raise NotImplementedError('You need to override this function')

    def server_status(self, timeout: int):
        """
        Provide server status. When cmd !='version', provide 'OK'
        should be implemented

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        raise NotImplementedError('You need to override this function')

    def preload_collection(self, table_name: str, timeout: int):
        """
        load table to memory cache in advance
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """

        raise NotImplementedError('You need to override this function')

    def describe_index(self, collection_name: str, timeout: int):
        """
        Show index information
        should be implemented

        :param collection_name: target collection name.
        :type collection_name: str

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

            TableSchema: table detail information

        :rtype: (Status, TableSchema)
        """

        raise NotImplementedError('You need to override this function')

    def drop_index(self, collection_name: str, timeout: int):
        """
        Show index information
        should be implemented

        :param collection_name: target collection name.
        :type collection_name: str

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """

        raise NotImplementedError('You need to override this function')
