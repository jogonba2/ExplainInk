from pathlib import Path
from typing import Dict
from setuptools import find_packages, setup

VERSION: Dict[str, str] = {}

with open("explainink/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

install_requires = [
    "typing_extensions==4.12.2",
    "transformers==4.46.3",
    "datasets==2.19.1",
    "torch==2.5.1",
    "accelerate==0.29.2",
    "scikit-learn==1.4.2",
    "typer==0.9.4",
    "spacy==3.7.4",
]


EXTRAS_REQUIRES: Dict[str, str] = {
    "data-based": ["bertopic==0.16.0", "keybert==0.8.4"],
    "captum": ["captum==0.7.0"],
    "notebooks": ["ipython>=8.12.3"],
    "shap": ["shap==0.45.0"],
    "adlfs": ["adlfs==2024.2.0"],
    "symanto-dec": ["sentence_transformers>=3.3.1", "symanto_dec", "azure-storage-blob", "optimum[onnxruntime]"],
    "networkx": ["networkx==3.3"],
    "symanto-fslapi-client": ["symanto-fslapi-client"]
}

EXTRAS_REQUIRES["all"] = sum(EXTRAS_REQUIRES.values(), [])

with Path("dev-requirements.txt").open("tr") as reader:
    EXTRAS_REQUIRES["dev"] = [line.strip() for line in reader]

setup(
    version=VERSION["VERSION"],
    name="explainink",
    description="Package for efficient feature attribution.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Symanto Research GmbH",
    author_email="jose.gonzalez@symanto.com",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=EXTRAS_REQUIRES,
    entry_points={"console_scripts": ["explainink=explainink.cli:app"]},
    python_requires=">=3.10.0",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
    ],
    license_files=[
        "LICENSE",
    ],
)
