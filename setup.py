from setuptools import setup

setup(
    name="dataframe-interpolator",
    version="0.22.12.31",
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
