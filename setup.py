import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="implicit_seg",
    version="0.0.1",
    author="Ruilong Li",
    author_email="ruilongl@usc.edu",
    description="3D & 2D segmentation via implicit function.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liruilong940607/implicit_seg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
