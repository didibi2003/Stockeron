from setuptools import setup, find_packages

setup(
    name='stock-forecast-app',
    version='1.0.0',
    packages=find_packages(),
    python_requires='>=3.7, <3.11',
    install_requires=[
        'streamlit',
        'numpy',
        'pandas',
        'matplotlib',
        'yfinance',
        'scikit-learn',
        'keras'
    ],
    author='Your Name',
    author_email='your@email.com',
    description='Stock Forecasting App',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
