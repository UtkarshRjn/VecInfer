from .Singleton import Singleton
from typing import Callable

class GeneratorBase:
    elem: any
    ops: Callable
    
    def __init__(self, elem: any, ops: Callable):
        self.elem = elem
        self.ops = ops
    
    def __call__(self):
        self.elem = self.ops(self.elem)
        return self.elem
    
class LayerIdxGenerator(GeneratorBase, metaclass=Singleton):
    def __init__(self):
        super().__init__(-1, lambda x: x + 1)

class NodeIdxGenerator(GeneratorBase, metaclass=Singleton):
    def __init__(self):
        super().__init__(-1, lambda x: x + 1)
