#!/usr/bin/env python3
"""
Setup script for video personalization pipeline
"""

from setuptools import setup, find_packages

setup(
    name="video-personalization",
    version="1.0.0",
    description="Automated video personalization with lip sync",
    author="Sankalp Thakur",
    packages=find_packages(),
    package_dir={'': '.'},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'flask>=2.3.0',
        'requests>=2.31.0',
        'tqdm>=4.65.0',
    ],
    entry_points={
        'console_scripts': [
            'personalize-video=src.core.pipeline:main',
        ],
    },
)