import numpy as np
import re
import os
from typing import Generator
from .Timer import tprint

def partition_generator(k, b):
    total_size = 256 * k
    # 计算每份的平均大小
    average_size = total_size // b
    # 计算最后一份的大小
    last_size = total_size - (b - 1) * average_size
    
    # 生成器函数
    for i in range(b):
        if i == b - 1:
            # 如果是最后一份，则大小为 last_size
            yield last_size
        else:
            # 其他份的大小都是 average_size
            yield average_size

def read_fvecs(filename):
    with open(filename, 'rb') as f:
        vecs = []
        while True:
            data = f.read(4)
            if len(data) < 4:
                break
            d = int.from_bytes(data, 'little')
            vec = np.frombuffer(f.read(d * 4), dtype=np.float32)
            vecs.append(vec)
        return np.array(vecs)

def write_fvecs(filename, vecs, mode='ab'):
    # TODO: async write
    if mode == 'ab' and not os.path.exists(filename):
        mode = 'wb'
    with open(filename, mode) as f:
        for vec in vecs:
            d = len(vec)
            f.write(np.int32(d).tobytes())
            f.write(vec.astype(np.float32).tobytes())

# async def write_fvecs_async(filename, vecs, mode='ab'):
#     async with aiofiles.open(filename, mode) as f:
#         for vec in vecs:
#             d = len(vec)
#             bytes_d = np.int32(d).tobytes()
#             bytes_vec = vec.astype(np.float32).tobytes()
#             await f.write(bytes_d)
#             await f.write(bytes_vec)

def natural_sort_key(s):
    match = re.search(r'_(\d+)_(\d+)\.fvecs', s)
    if match:
        layer = int(match.group(1))
        h = int(match.group(2))
    return layer * 1000 + h # 总不会有1000个头吧?

def read_fvecs_batch(root, pattern='key\w+.fvecs', pg=None):
    files = os.listdir(root)
    files = [f for f in files if re.match(pattern, f)]
    files.sort(key=natural_sort_key)

    # print(f"Reading {len(files)} files from {root}...")
    # print(f"Files: {files}")

    vecs = []
    if pg is None:
        pg = [None] * len(files)
    # TODO: parallelize this, extremely slow if pg is not None
    for fileName, sample_size in zip(files, list(pg)):
        with open(root / fileName, 'rb') as f:
            local_vecs = []
            while True:
                data = f.read(4)
                if len(data) < 4:
                    if sample_size is not None:
                        if len(local_vecs) > sample_size:
                            # tprint(f"Downsampled {len(local_vecs)} to {sample_size}")
                            idx = np.random.choice(len(local_vecs), sample_size, replace=False)
                            vecs.extend(np.array(local_vecs)[idx])
                    else:
                        vecs.extend(local_vecs)
                    break
                d = int.from_bytes(data, 'little')
                vec = np.frombuffer(f.read(d * 4), dtype=np.float32)
                local_vecs.append(vec)
    return np.array(vecs)


def sample_fvecs(root, pattern, pg, save_path=None):
    files = os.listdir(root)
    files = [f for f in files if re.match(pattern, f)]

    with open(root / files[0], 'rb') as f:
        data = f.read(4)
        dim = int.from_bytes(data, 'little')

    size_per_vec = 4 * (1 + dim)
    from tqdm import tqdm

    vecs = []
    for fileName, sample_size in tqdm(zip(files, list(pg))):
        with open(root / fileName, 'rb') as f:
            # calculate the number of vectors in the file
            f.seek(0, 2)
            total_size = f.tell()
            f.seek(0)
            total_vecs = total_size // size_per_vec

            idx = np.random.choice(total_vecs, sample_size, replace=False)
            for i in idx:
                f.seek(i * size_per_vec)
                data = f.read(4)
                d = int.from_bytes(data, 'little')
                vec = np.frombuffer(f.read(d * 4), dtype=np.float32)
                vecs.append(vec)
        
    if save_path is None:
        if 'key' in pattern:
            filename = root / 'key_sampled.fvecs'
        elif 'value' in pattern:
            filename = root / 'value_sampled.fvecs'
        else:
            filename = root / 'sampled.fvecs'
            print(f'Unknown pattern')
    else:
        filename = save_path

    print(f'Saving to {filename}')
    write_fvecs(filename, vecs, 'wb')

if __name__ == "__main__":
    from pathlib import Path
    root = './kv_samples/llama-2-7b/wikitext-103-raw-v1/'
    root = Path(root)

    nbits_key = 8
    nbits_value = 8
    M_key = 64
    M_value = 64
    nh = 32
    n_layer = 32
    
    sample_fvecs(root, r'key(?!.*_sampled)\w+\.fvecs', partition_generator(2*2**nbits_key, nh*n_layer))
    sample_fvecs(root, r'value(?!.*_sampled)\w+\.fvecs', partition_generator(2*2**nbits_value, nh*n_layer))