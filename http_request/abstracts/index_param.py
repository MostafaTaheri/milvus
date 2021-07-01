import json
from typing import List

from milvus import ParamError
from http_request.constants.index_type import IndexType


class IndexParam:
    """
    Index Param

    :type  table_name: str
    :param table_name: (Required) name of table

    :type  index_type: IndexType
    :param index_type: (Required) index type, default = IndexType.INVALID

        `IndexType`: 0-invalid, 1-flat, 2-ivflat, 3-IVF_SQ8, 4-MIX_NSG

    :type  nlist: int64
    :param nlist: (Required) num of cell

    """
    def __init__(self, collection_name: str, index_type: IndexType,
                 params: List):

        if collection_name is None:
            raise ParamError('Collection name can\'t be None')
        collection_name = str(collection_name) if not isinstance(
            collection_name, str) else collection_name

        if isinstance(index_type, int):
            index_type = IndexType(index_type)
        if not isinstance(index_type,
                          IndexType) or index_type == IndexType.INVALID:
            raise ParamError(
                'Illegal index_type, should be IndexType but not IndexType.INVALID'
            )

        self._collection_name = collection_name
        self._index_type = index_type

        if not isinstance(params, dict):
            self._params = json.loads(params)
        else:
            self._params = params

    def __str__(self):
        attr_list = [
            '%s=%r' % (key.lstrip('_'), value)
            for key, value in self.__dict__.items()
        ]
        return '(%s)' % (', '.join(attr_list))

    def __repr__(self):
        attr_list = [
            '%s=%r' % (key, value) for key, value in self.__dict__.items()
        ]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    @property
    def collection_name(self):
        return self._collection_name

    @property
    def index_type(self):
        return self._index_type

    @property
    def params(self):
        return self._params
