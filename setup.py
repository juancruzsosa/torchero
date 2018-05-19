from setuptools import setup, find_packages

setup(name='torchtrainer',
      version='0.0.2',
      description='A pluggable & extensible trainer for pytorch',
      url='http://github.com/nirvguy/torchtrainer',
      author='Juan Cruz Sosa',
      author_email='nirvguy@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'torch',
          'pyyaml'
      ],
      extra_requires={
        'monitor training/validation with progress bar support' : ['tqdm']
      },
      zip_safe=False)
