from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mypkg',
    ext_modules=[
        CUDAExtension('mypkg', [
            'hello-extension.cpp',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
