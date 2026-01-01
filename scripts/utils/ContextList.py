class ContextList(list):

    def __enter__(self):
        for item in self:
            item.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for item in self:
            item.__exit__(exc_type, exc_value, traceback)
        return False