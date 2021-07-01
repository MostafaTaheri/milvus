class PartitionParam:
    def __init__(self, collection_name, tag):
        self.collection_name = collection_name
        self.tag = tag

    def __repr__(self):
        attr_list = [
            '%s=%r' % (key, value) for key, value in self.__dict__.items()
        ]
        return '(%s)' % (', '.join(attr_list))
