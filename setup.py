from setuptools import setup

setup(
    name='linc_cv',
    version='1.0.0',
    packages=['linc_cv'],
    entry_points={
        'console_scripts': [
            'linc_cv = linc_cv.__main__:main'
        ]
    })
