from setuptools import setup, find_packages, find_namespace_packages

setup(
    name="averle",
    version="0.1",
    install_requires=["gymnasium", "numpy", "pandas", "torch"],
    packages=find_namespace_packages(),
)
