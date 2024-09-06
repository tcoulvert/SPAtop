from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="SPAtop",
    author="Javier Duarte, Billy Li, Melissa Quinnan, Thomas Sievert",
    url="https://github.com/tcoulvert/SPAtop",
    license="MIT",
    install_requires=[
        "coffea",
        "spanet",
        "numpy",
        "xxhash",
        "vector",
    ],
)
