class QueryResult:
    def __init__(self, _id, _distance):
        self.id = _id
        self.distance = _distance

    def __str__(self):
        return "Result(id={}, distance={})".format(self.id, self.distance)
