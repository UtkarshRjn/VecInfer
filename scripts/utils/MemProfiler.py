import torch

class MemProfiler:
    """
    example usage:
        MemProfiler.report_cuda_memory(locals())
    """

    def __init__(self):
        pass

    @staticmethod
    def report_cuda_memory(namespace):
        for name, obj in namespace.items():
            if torch.is_tensor(obj):
                # size in MiB
                print(f"{name}: {obj.element_size() * obj.nelement() / 1024 / 1024} MiB, Device: {obj.device}, shape: {obj.shape}")
            elif isinstance(obj, torch.nn.Module):
                print(f"{name}: is a torch.nn.Module, not yet implemented mem profiling")
