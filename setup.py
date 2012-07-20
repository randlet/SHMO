from setuptools import setup

setup(
    author = "Randle Taylor",
    author_email = "randle.taylor@gmail.com",
    url = "randlet.com",
    name='SHMO',
    version='0.1',
    packages=['shmo',],
    test_suite= 'nose.collector',
    tests_require=["nose"],
    setup_requires=["nose>=1.0"],
    license='BSD',
    long_description=open('README.rst').read(),
)
