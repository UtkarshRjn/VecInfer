class LazyLoader:
    def __init__(self, loader_fn, *args, **kwargs):
        """
        A generic lazy loader.

        :param loader_fn: A callable that will be executed lazily.
        :param args: Positional arguments to pass to the loader function.
        :param kwargs: Keyword arguments to pass to the loader function.
        """
        self._loader_fn = loader_fn
        self._loader_args = args
        self._loader_kwargs = kwargs
        self._loaded_object = None

    def _load(self):
        if self._loaded_object is None:
            self._loaded_object = self._loader_fn(*self._loader_args, **self._loader_kwargs)

    def __call__(self, *args, **kwargs):
        self._load()
        return self._loaded_object(*args, **kwargs)

    def __getattr__(self, name):
        self._load()
        return getattr(self._loaded_object, name)
