from setuptools import setup, find_packages

setup(
    name='geosketch',
    version='1.0',
    description='Geometry-preserving random sampling',
    url='https://github.com/brianhie/geosketch',
    download_url='https://github.com/brianhie/geosketch/archive/v1.0-beta.tar.gz',
    packages=find_packages(exclude=['bin', 'conf', 'data', 'target', 'R']),
    install_requires=[
        'fbpca>=1.0',
        'numpy>=1.12.0',
        'scikit-learn>=0.20rc1',
    ],
    author='Brian Hie',
    author_email='brianhie@mit.edu',
    license='MIT'
)
