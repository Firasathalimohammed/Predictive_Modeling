from setuptools import setup, find_packages

setup(
    name="src.Car_analysis",
    version="0.1",
    packages=find_packages(),
    description="A Python package for analyzing Car Dataset",
    author="Firasath",
    author_email="fmohamm1@mail.yu.edu",
    url="https://github.com/Firasathalimohammed/Predictive_Modeling",
    install_requires=["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"],
)
