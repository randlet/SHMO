from distutils.core import setup

setup(
    name='SHMO',
    version='0.1dev',
    packages=['shmo',],
    test_suite='shmo.test',
    license='BSD',
    long_description=open('README.rst').read(),
)