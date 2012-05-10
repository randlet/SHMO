from setuptools import setup

setup(
    name='SHMO',
    version='0.1dev',
    packages=['shmo',],
    test_suite= 'nose.collector',
    tests_require=["nose"],
    setup_requires=["nose>=1.0"],
    license='BSD',
    long_description=open('README.rst').read(),
)
