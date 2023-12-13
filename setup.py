from setuptools import setup, find_packages

setup(
    name="gbiz_torch",
    version="0.0.6.2",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "scikit-learn>=1.2.2",
    ],

    author="Haowen Wang",
    author_email="haowenw98@gmail.com",
    description="general deep algorithm for business, marketing and advertisement",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/whw199833/gbiz_torch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
