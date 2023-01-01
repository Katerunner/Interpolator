from setuptools import setup

from dataframe_interpolator import version

setup(
    name="dataframe-interpolator",
    version=version,
    author="Katerunner",
    author_email="teliuk2@gmail.com",
    description="Script that interpolates NaN in all pandas dataset that contains numerical values",
    packages=['dataframe_interpolator'],
    python_requires=">=3.6",
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'scikit-learn',
        'matplotlib'
    ]
)
