# @Author            : FederalLab
# @Date              : 2021-09-25 16:57:29
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:57:29
# Copyright (c) FederalLab. All rights reserved.
"""python setup.py sdist bdist_wheel python -m twine upload dist/*"""
from setuptools import find_packages, setup


def get_version():
    version_file = 'openfed/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


with open('requirements.txt', 'r', encoding='utf-8') as fh:
    install_requires = fh.read()

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name='openfed',
    version=get_version(),
    author='FederalLab',
    author_email='densechen@foxmail.com',
    description='OpenFed: A PyTorch Library for Federated Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/FederalLab/OpenFed',
    download_url='https://github.com/FederalLab/OpenFed/archive/main.zip',
    packages=find_packages(),
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Security',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='MIT License',
    keywords='federated learning',
)
