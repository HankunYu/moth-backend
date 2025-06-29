#!/usr/bin/env python3
"""
Setup script for Moth Backend
"""

from setuptools import setup, find_packages
import pathlib

# Read the contents of README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "INSTALL.md").read_text(encoding="utf-8")

setup(
    name="moth-backend",
    version="1.0.0",
    description="Genetic algorithm-based moth breeding and lifecycle management system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Moth Backend Team",
    author_email="",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="genetic-algorithm, image-processing, breeding, lifecycle-management",
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "moth-controller=moth_controller:main",
        ],
    },
    project_urls={
        "Bug Reports": "",
        "Source": "",
    },
)