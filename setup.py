from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="nanofts",
    version="0.1.0",
    description="A lightweight full-text search library for Python",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "msgpack",
        "pyroaring",
    ],
    extras_require={
        'pandas': ['pandas>=1.0.0'],
        'polars': ['polars>=0.20.0'],
        'pyarrow': ['pyarrow>=14.0.0'],
        'all': [
            'pandas>=1.0.0',
            'polars>=0.20.0',
            'pyarrow>=14.0.0',
        ],
    },
    python_requires=">=3.9",
) 