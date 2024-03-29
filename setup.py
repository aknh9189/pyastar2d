import numpy
from setuptools import setup, find_packages, Extension

astar_module = Extension(
    name = 'pyastar2d.astar',
    sources=['src/cpp/astar.cpp', 'src/cpp/experimental_heuristics.cpp'],
    include_dirs=[numpy.get_include(), 'src/cpp'],
    extra_compile_args=["-O3", "-Wall", "-shared", "-fpic"],
)

djikstra_module = Extension(
    name = 'pyastar2d.djikstra',
    sources=['src/cpp/djikstra.cpp'],
    include_dirs=[numpy.get_include(), 'src/cpp'],
    extra_compile_args=["-O3", "-Wall", "-shared", "-fpic"],
)

setup_args = dict(
    packages = find_packages(where="src"),
    package_dir = {"": "src"},
    ext_modules = [astar_module, djikstra_module]
)
setup(
    **setup_args
)

