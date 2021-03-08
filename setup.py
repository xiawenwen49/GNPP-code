from setuptools import setup
from setuptools import find_packages

setup(name='tgnn',
      version='0.1',
      description='tgnn for temporal graphs applications',
      author='Xia Wenwen',
      author_email='xiawenwen@sjtu.edu.cn',
      license='MIT',
      package_dir={'': 'src/'},
      packages=['dtgnn', 'tge'],
      zip_safe=False)

# packages=find_packages('src/python'),