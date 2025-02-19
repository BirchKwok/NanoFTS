from setuptools import setup, find_packages

setup(
    name="nanofts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "msgpack",
        "pyroaring",
    ],
    python_requires=">=3.9",
) 