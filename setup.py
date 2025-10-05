from setuptools import setup, find_packages

setup(
    name="eTIS_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "seaborn",
        "matplotlib",
        "tqdm",
        "biopython",
        "scipy"
    ],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "eTIS_model=eTIS_model.cli:main",  # if you have a CLI
        ],
    },
)