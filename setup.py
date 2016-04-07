__author__ = 'giacomov'

#!/usr/bin/env python

from setuptools import setup

setup(
    name="lclike",

    packages=['lclike'],

    scripts= ['lclike/lclike.py'],

    #data_files=[('astromodels/data/functions', glob.glob('astromodels/data/functions/*.yaml'))],

    version='0.1',

    description="Fit in the counts' space of Gamma-Ray Bursts detected by Fermi/LAT",

    author='Giacomo Vianello, Jarred Gillette',

    author_email='giacomo.vianello@gmail.com',

    url='https://github.com/giacomov/lclike',

    download_url='https://github.com/giacomov/lclike/archive/v0.1',

    keywords=['Likelihood', 'Models', 'fit', 'LAT'],

    classifiers=[],

    install_requires=[
        'numpy >= 1.6',
        'iminuit']

)
