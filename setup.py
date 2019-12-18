from setuptools import setup


setup(name='saxs_analyzer',
      version='0.1.1',
      description='Some plotting for comparing predicted RNA structures with SAXS',
      author='Bernhard Thiel',
      author_email='thiel@tbi.univie.ac.at',
      py_modules=["saxs_analyzer"],
      install_requires=[
        "forgi"]
     )
