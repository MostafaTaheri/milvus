from .query_result import QueryResult


class TopKQueryResult:
    def __init__(self, raw_source, **kwargs):
        self._raw = raw_source
        self._nq = 0
        self._topk = 0
        self._results = []

        self.__index = 0

        self._unpack(self._raw)

    def _unpack(self, raw_resources):
        js = raw_resources.json()
        self._nq = js["num"]

        for row_result in js["result"]:
            row_ = [
                QueryResult(int(result["id"]), float(result["distance"]))
                for result in row_result if float(result["id"]) != -1
            ]

            self._results.append(row_)

    @property
    def shape(self):
        return len(self._results), len(
            self._results[0]) if len(self._results) > 0 else 0

    def __len__(self):
        return len(self._results)

    def __getitem__(self, item):
        return self._results.__getitem__(item)

    def __iter__(self):
        return self

    def __next__(self):
        while self.__index < self.__len__():
            self.__index += 1
            return self.__getitem__(self.__index - 1)

        self.__index = 0
        raise StopIteration()

    def __repr__(self):
        lam = lambda x: "(id:{}, distance:{})".format(x.id, x.distance)

        if self.__len__() > 5:
            middle = ''

            ll = self[:3]
            for topk in ll:
                if len(topk) > 5:
                    middle = middle + " [ %s" % ",\n   ".join(
                        map(lam, topk[:3]))
                    middle += ",\n   ..."
                    middle += "\n   %s ]\n\n" % lam(topk[-1])
                else:
                    middle = middle + " [ %s ] \n" % ",\n   ".join(
                        map(lam, topk))

            spaces = """        ......
                    ......"""

            ahead = "[\n%s%s\n]" % (middle, spaces)
            return ahead

        str_out_list = []
        for i in range(self.__len__()):
            str_out_list.append("[\n%s\n]" % ",\n".join(map(lam, self[i])))

        return "[\n%s\n]" % ",\n".join(str_out_list)
