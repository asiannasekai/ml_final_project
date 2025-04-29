from setuptools import setup, find_packages

setup(
    name="mycelial_router",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "networkx",
        "matplotlib",
    ],
) 