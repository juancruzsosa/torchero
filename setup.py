from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='torchero',
      version='0.0.4',
      description='A pluggable & extensible trainer for pytorch',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/juancruzsosa/torchero',
      author='Juan Cruz Sosa',
      author_email='juancruzsosa.92@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'torch',
          'pyyaml',
          'tqdm',
          'requests',
          'matplotlib',
          'Pillow'
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      zip_safe=False)
