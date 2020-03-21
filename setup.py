from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules=[
    CUDAExtension('seg_renderer.cuda.reconstruction', [
        'soft_renderer/cuda/reconstruction_cuda.cpp',
        'soft_renderer/cuda/reconstruction_cuda_kernel.cu',
        ]),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'tqdm', 'imageio']

setup(
    description='',
    author='Ruilong Li',
    author_email='',
    license='MIT License',
    version='0.0.1',
    name='seg_renderer',
    packages=['seg_renderer', 'seg_renderer.cuda', 'seg_renderer.functional'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)