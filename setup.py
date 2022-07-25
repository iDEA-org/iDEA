import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="iDEA-latest",
    version="0.1.3",
    author="Jack Wetherell",
    author_email="jack.wetherell@gmail.com",
    description="interacting Dynamic Electrons Approach (iDEA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iDEA-org/iDEA",
    project_urls={
        "Bug Tracker": "https://github.com/iDEA-org/iDEA/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "iDEA"},
    packages=setuptools.find_packages(where="iDEA"),
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.22.3',
        'scipy>=1.8.0',
        'matplotlib>=3.5.1',
        'jupyterlab>=3.3.2',
        'tqdm>=4.64.0',
        'black>=22.3.0',
        'autoflake>=1.4.0',
        'build>=0.7.0',
        'twine>=4.0.0',
    ],
)