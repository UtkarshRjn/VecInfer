'''
This module is guided by AOP(Aspect Oriented Programming) principles.
It provides a way to inject a function into a class method, and then restore the original method.

"Patching an object is more robust than patching source codes, and is more flexible than subclassing." -- Xp

'''

from typing import Any, Callable

class Injector:

    def __init__(self, node, attr_name: str, other_attr):
        self.node = node
        self.attr_name = attr_name
        self._attr = getattr(node, attr_name, None)
        self.other_attr = other_attr
    
    def __enter__(self):
        setattr(self.node, self.attr_name, self.other_attr)
        return self.node
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._attr is None:
            delattr(self.node, self.attr_name)
        else:
            setattr(self.node, self.attr_name, self._attr)
        return False
    
class PrefixInjector:
    
    def __init__(self, node, attr_name: str, prefix: Callable):
        self.node = node
        self.attr_name = attr_name
        self.prefix = prefix
        
    def __enter__(self):
        self._attr = getattr(self.node, self.attr_name)

        def prefixed(*args, **kwargs):
            self.prefix(*args, **kwargs)
            return self._attr(*args, **kwargs)
        self.prefixed = prefixed

        setattr(self.node, self.attr_name, self.prefixed)
        return self.node
    
    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.node, self.attr_name, self._attr)
        return False 
    
class PreSuffixInjector:

    def __init__(self, node, attr_name: str, prefix: Callable, suffix: Callable):
        self.node = node
        self.attr_name = attr_name
        self.prefix = prefix
        self.suffix = suffix

    def __enter__(self):
        self._attr = getattr(self.node, self.attr_name)

        def altered(*args, **kwargs):
            if self.prefix is not None:
                self.prefix(*args, **kwargs)
            res = self._attr(*args, **kwargs)
            if self.suffix is not None:
                self.suffix(*args, **kwargs)
            return res
        self.altered = altered

        setattr(self.node, self.attr_name, self.altered)
        return self.node
    
    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.node, self.attr_name, self._attr)
        return False