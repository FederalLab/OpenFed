  
'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''
import os

from setuptools import find_packages, setup

from openfed import __version__

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name="openfed",
    version=__version__,
    author="densechen",
    author_email="densechen@foxmail.com",
    description="OpenFed: A PyTorch Library for Federated Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FederalLab/OpenFed",
    download_url = 'https://github.com/FederalLab/OpenFed/archive/main.zip',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT License",
    python_requires='>=3.6',
)