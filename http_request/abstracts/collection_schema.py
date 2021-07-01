from http_request.constants import MetricType
from milvus.check import check_pass_param


class CollectionSchema:
    def __init__(self, collection_name: str, dimension: int,
                 index_file_size: int, metric_type: MetricType):
        """
        Table Schema

        :type  table_name: str
        :param table_name: (Required) name of table

            `IndexType`: 0-invalid, 1-flat, 2-ivflat, 3-IVF_SQ8, 4-MIX_NSG

        :type  dimension: int64
        :param dimension: (Required) dimension of vector

        :type  index_file_size: int64
        :param index_file_size: (Optional) max size of files which store index

        :type  metric_type: MetricType
        :param metric_type: (Optional) vectors metric type

            `MetricType`: 1-L2, 2-IP

        """
        check_pass_param(collection_name=collection_name,
                         dimension=dimension,
                         index_file_size=index_file_size,
                         metric_type=metric_type)

        self.collection_name = collection_name
        self.dimension = dimension
        self.index_file_size = index_file_size
        self.metric_type = metric_type

    def __repr__(self):
        attr_list = [
            '%s=%r' % (key, value) for key, value in self.__dict__.items()
        ]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))
