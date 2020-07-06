from setuptools import setup, find_packages

setup(name='torchero',
      version='0.1',
      description='A pluggable & extensible trainer for pytorch',
      url='http://github.com/nirvguy/torchero',
      author='Juan Cruz Sosa',
      author_email='nirvguy@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'torch',
          'pyyaml'
      ],
      extra_requires={
        'monitor training/validation with progress bar support' : ['tqdm'],
        'remote monitor callback': ['requests']
      },
      zip_safe=False)
