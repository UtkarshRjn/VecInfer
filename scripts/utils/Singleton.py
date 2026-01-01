from abc import ABCMeta

class Singleton(type):
    '''
    Singleton metaclass, use example:

    class MyClass(metaclass=Singleton):
        pass
    
    my_class = MyClass()

    Note that this is not multi-process safe.
    '''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    @classmethod
    def has_instance(cls):
        return len(cls._instances) > 0
    
    @classmethod
    def clear_instance(cls):
        for instance in cls._instances.values():
            del instance
        cls._instances = {}

class SingletonABCMeta(Singleton, ABCMeta):
    pass

class NamedSingleton(type):
    '''
    Named Singleton metaclass, ensures that each specific name can only have one instance.
    Use example:

    class NamedClass(metaclass=NamedSingleton):
        def __init__(self, name):
            self.name = name

    a = NamedClass("instance1")
    b = NamedClass("instance2")
    c = NamedClass("instance1")

    print(a is b)  # False, different names
    print(a is c)  # True, same name
    '''

    _instances = {}

    def __call__(cls, name, *args, **kwargs):
        if name not in cls._instances:
            cls._instances[name] = super(NamedSingleton, cls).__call__(name, *args, **kwargs)
        return cls._instances[name]


