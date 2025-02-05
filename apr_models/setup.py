# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name='apr_models',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pyapr'
    ],
    description='APR-CNN models',
    author='Joel Jonsson',
)

