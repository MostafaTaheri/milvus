from enum import IntEnum


class MetricType(IntEnum):
    INVALID = 0
    L2 = 1
    IP = 2
    # Only supported for byte vectors
    HAMMING = 3
    JACCARD = 4
    TANIMOTO = 5

    SUBSTRUCTURE = 6
    SUPERSTRUCTURE = 7

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name_)

    def __str__(self):
        return self._name_
