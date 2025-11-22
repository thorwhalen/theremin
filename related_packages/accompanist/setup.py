"""Setup configuration for accompanist package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="accompanist",
    version="0.1.0",
    author="Theremin Project",
    description="Music accompaniment tools - chord progressions and MIDI utilities",
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
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No hard dependencies for basic functionality
    ],
    extras_require={
        "midi": ["mido>=1.2.0"],
        "all": ["mido>=1.2.0"],
    },
)
