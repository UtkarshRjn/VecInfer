from time import time, time_ns
from .Singleton import Singleton, NamedSingleton
import torch

class BaseTimer:
    disabled = False
    def __init__(self, init_start=False):
        self.init_start = init_start
        self.reset()

    def start(self):
        if self.disabled: return

        self._start = time()

    def pause(self):
        if self.disabled: return

        self.duration += time() - self._start
        self.paused = True

    def tick(self):
        if self.disabled: return self.duration

        if self.paused:
            return self.duration
        return time() - self._start + self.duration
    
    def reset(self):
        self.duration = 0
        if self.init_start:
            self.paused = False
            self.start()
        else:
            self.paused = True

    def __str__(self):
        return f'[{self.tick():.6f}s] '

    def __repr__(self) -> str:
        return str(self)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.pause()
    
    def time_this(self, func):
        '''
        Can be used as a decorator to time a function.
        Example:
            timer = BaseTimer()

            @timer.time_this
            def foo():
                pass
            
            [foo() for _ in range(10)]
            print(timer)
        '''
        def wrapper(*args, **kwargs):
            with self:
                res = func(*args, **kwargs)
            return res
        return wrapper

class Timer(BaseTimer, metaclass=NamedSingleton):
    def __init__(self, name='default'):
        self.name = name
        super().__init__()

    def __str__(self):
        return f'[{self.name} {self.tick():.5f}s] '

class UniTimer(BaseTimer, metaclass=Singleton):
    pass

class BaseTicker:
    def __init__(self):
        self.ticks = []
    
    def tick(self, *args, **kwargs):
        # torch.cuda.synchronize()
        self.ticks.append(time_ns())

    def clear(self):
        self.ticks = []

    @property
    def intervals(self):
        return [self.ticks[i] - self.ticks[i-1] for i in range(1, len(self.ticks))]
    
class Ticker(BaseTicker, metaclass=NamedSingleton):
    def __init__(self, name='default'):
        self.name = name
        super().__init__()

def tprint(*args, **kwargs):
    print(Timer(''), *args, **kwargs)