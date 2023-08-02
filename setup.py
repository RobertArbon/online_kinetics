import setuptools

setuptools.setup(
    name="celerity",
    version="0.1",
    description="Creates deep MSMs using accelerated training methods",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    entry_points={
        'console_scripts': [
            'celerity = celerity.api:cli',
        ],
    },
    python_requires=">=3.10",
)