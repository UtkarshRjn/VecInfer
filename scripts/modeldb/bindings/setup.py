import os
import shutil
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from itertools import product

################ META PROGRAMMING ################

float_list = ['f16']
uint_list = ['u8']
Ns_list = [1, 2, 4, 8, 16, 32, 64]
d_list = [64, 128, 256]
M_key_list = [16, 32, 64]
M_value_list = [16, 32, 64]
C_list = [256]
Lt_list = [64, 128, 256]

# generate cuda file
cuda_template = "Interface.template.cu"
cuda_outfile  = "Interface.cu"
with open(cuda_template, "r") as fin, open(cuda_outfile, "w") as fout:
    for line in fin:
        fout.write(line)
    for f, u, Ns, d, M_key, M_value, C, Lt in product(float_list, uint_list, Ns_list, d_list, M_key_list, M_value_list, C_list, Lt_list):
        fout.write(f"register_flash_decoding_allocated_buffer({f}, {u}, {Ns}, {Lt}, {d}, {M_key}, {M_value}, {C});\n")

# generate bindings.cpp
cpp_template = "bindings.template.cpp"
cpp_outfile  = "bindings.cpp"
with open(cpp_template, "r") as fin, open(cpp_outfile, "w") as fout:
    for line in fin:
        fout.write(line)
    for f, u, Ns, d, M_key, M_value, C, Lt in product(float_list, uint_list, Ns_list, d_list, M_key_list, M_value_list, C_list, Lt_list):
        fout.write(f"declare_flash_decoding_allocated_buffer({f}, {u}, {Ns}, {Lt}, {d}, {M_key}, {M_value}, {C});\n")
    fout.write(r"PYBIND11_MODULE(bindings, m) {" + "\n")
    for f, u, Ns, d, M_key, M_value, C, Lt in product(float_list, uint_list, Ns_list, d_list, M_key_list, M_value_list, C_list, Lt_list):
        fout.write(f"    m.def(\"flash_decoding_allocated_buffer_{f}{u}_Ns{Ns}Lt{Lt}d{d}M_key{M_key}M_value{M_value}C{C}\", &flash_decoding_allocated_buffer_{f}{u}_Ns{Ns}Lt{Lt}d{d}M_key{M_key}M_value{M_value}C{C});\n")
    fout.write(r"}" + "\n")

##################################################

# class CustomBuild(BuildExtension):
#     def run(self):
#         # 正常 build
#         super().run()

#         # 找到中间目录里的 .ptx 文件并复制出来
#         build_dir = self.build_temp
#         out_dir = os.path.join(os.path.dirname(__file__), "ptx")
#         os.makedirs(out_dir, exist_ok=True)

#         for root, _, files in os.walk(build_dir):
#             for f in files:
#                 if f.endswith(".ptx"):
#                     src = os.path.join(root, f)
#                     dst = os.path.join(out_dir, f)
#                     print(f"[PTX] Saving {src} -> {dst}")
#                     shutil.copyfile(src, dst)

setup(
    name='bindings',
    ext_modules=[
        CUDAExtension(
            name='bindings',
            sources=[cuda_outfile, cpp_outfile],
            include_dirs=['./', './core'],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-std=c++17",
                    "-O3",
                    "-Xcompiler", "-fPIC",
                    "-arch=sm_90",
                    "--use_fast_math",
                    "-Xptxas", "-v",
                    "-keep",   
                ],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    install_requires=[
        'torch',
        'pybind11',
    ]
)
