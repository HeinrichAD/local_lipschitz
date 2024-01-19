from setuptools import setup


# read version file
# exec(open("src/version.py").read())
__version__ = "0.0.1-dev1"


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="local_lipschitz",
    # author="uwaa-ndcl",
    # author_email="uwaa-ndcl@github.com",
    version=__version__,  # type: ignore # noqa F821
    description="Analytical Bounds on the Local Lipschitz Constants of ReLU Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/HeinrichAD/local_lipschitz",
    license_files=["LICENSE"],
    packages=["local_lipschitz"],
    package_dir={
        "local_lipschitz": "local_lipschitz",
    },
    # include_package_data=True,
    python_requires=">=3.7",
    install_requires=[  # TODO: add version numbers
        "torchextractor",
        "matplotlib",
        "numpy",
        "pillow",
        "torch",
        "torchvision",
        "tqdm",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        # "Topic :: Scientific/Engineering",
    ],
)
