from setuptools import setup

setup(name='graph_wavenet',
    version='0.1.0',
    description='Graph Wavenet Implementation',
    author='',
    url='https://github.com/ggaoshen/graph_wavenet',
    packages=['graph_wavenet'],
    install_requires=[
        "torch",
        "torch-geometric",
        "pandas",
        "numpy",
        "scipy",
        "networkx",
        "matplotlib",
        "argparse",
        "black"
    ]
)