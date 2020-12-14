import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

with open('VERSION', 'r') as f:
    version = f.read().strip()

setuptools.setup(
    name='aakr',
    version=version,
    license='MIT',
    description='Implementation of Auto Associative Kernel Regression (AAKR)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jesse Myrberg',
    author_email='jesse.myrberg@gmail.com',
    url='https://github.com/jmyrberg/aakr',
    keywords=['aakr', 'auto', 'associative', 'kernel', 'regression', 'anomaly', 'detection'],
    install_requires=[
        'numpy>=1.19.4',
        'pandas>=1.1.5',
        'scikit-learn>=0.23.2'
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-cov'],
        'docs': [
            'sphinx',
            'sphinx-gallery',
            'sphinx_rtd_theme',
            'numpydoc',
            'matplotlib'
        ]
    }
)