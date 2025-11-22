"""Setup configuration for synthflow package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="synthflow",
    version="0.1.0",
    author="Theremin Project",
    description="Dict-based synthesizer control - framework agnostic audio synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thorwhalen/theremin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "pyo": ["pyo>=1.0.0"],
        "effects": ["scipy>=1.5.0"],
        "all": ["pyo>=1.0.0", "scipy>=1.5.0"],
    },
)
