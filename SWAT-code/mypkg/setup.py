from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mypkg",
    ext_modules=[
        CUDAExtension(
            name="mypkg",
            sources=[
                "hello-extension.cpp",
            ],
            libraries=["cudnn"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
