from setuptools import setup, find_packages

setup(name='torchero',
      version='0.1',
      description='A pluggable & extensible trainer for pytorch',
      url='http://github.com/juancruzsosa/torchero',
      author='Juan Cruz Sosa',
      author_email='juancruzsosa.92@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'torch',
          'pyyaml',
          'tqdm',
          'requests'
      ],
      zip_safe=False)
