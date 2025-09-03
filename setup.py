import setuptools
import os
import sys
from pathlib import Path

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Version handling
try:
    version_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.append(version_path)
    from version import __version__
    package_name = __version__
except ImportError:
    package_name = "epibert"
    
def read_requirements(filename):
    """Read requirements from file"""
    req_file = Path(__file__).parent / filename
    if req_file.exists():
        with open(req_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Core requirements (minimal for base installation)
core_requirements = [
    'numpy>=1.23.0',
    'pandas>=2.0.0',
    'scipy>=1.10.0',
    'scikit-learn>=1.3.0',
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'h5py>=3.8.0',
    'pysam>=0.22.0',
    'pybedtools>=0.9.0',
    'einops>=0.8.0',
    'wandb>=0.15.0',
    'tqdm>=4.65.0',
    'logomaker>=0.8.0',
]

# Implementation-specific requirements
lightning_requirements = [
    'torch>=2.0.0',
    'pytorch-lightning>=2.0.0',
    'torchmetrics>=0.11.0',
    'torchvision>=0.15.0',
]

tensorflow_requirements = [
    'tensorflow>=2.12.0',
    'tensorflow-addons>=0.23.0',
    'tensorboard>=2.12.0',
]

# Additional optional requirements
extras_require = {
    'lightning': lightning_requirements,
    'tensorflow': tensorflow_requirements,
    'attribution': ['jax>=0.4.0', 'jaxlib>=0.4.0'],
    'analysis': ['kipoi>=0.8.0'],
    'dev': [
        'pytest>=7.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'mypy>=1.0.0',
        'jupyter>=1.0.0',
    ],
    'all': lightning_requirements + tensorflow_requirements + ['jax>=0.4.0', 'jaxlib>=0.4.0', 'kipoi>=0.8.0'],
}

setuptools.setup(
    name=package_name,
    version="0.2.0",  # Updated version
    author="N Javed, T Weingarten",
    author_email="javed@broadinstitute.org",
    description="EpiBERT: Multi-modal transformer for cell type-agnostic regulatory predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biubiu-1/EpiBERT",
    project_urls={
        "Bug Tracker": "https://github.com/biubiu-1/EpiBERT/issues",
        "Documentation": "https://github.com/biubiu-1/EpiBERT",
        "Source": "https://github.com/biubiu-1/EpiBERT",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=core_requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "epibert-setup=scripts.setup_environment:main",
            "epibert-validate=scripts.validate_setup:main",
            "epibert-train=scripts.train_model:main",
            "epibert-evaluate=scripts.evaluate_model:main",
        ],
    },
    keywords=[
        "genomics", "epigenomics", "machine learning", "deep learning", 
        "transformer", "attention", "regulatory genomics", "chromatin accessibility",
        "ATAC-seq", "RAMPAGE-seq", "PyTorch Lightning", "TensorFlow"
    ],
    zip_safe=False,
)
