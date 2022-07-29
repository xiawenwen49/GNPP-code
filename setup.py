from setuptools import setup
from setuptools import find_packages

setup(name='gnpp',
      version='0.1',
      description='GNPP for temporal graph prediction',
      author='Xia Wenwen',
      author_email='xiawenwen@sjtu.edu.cn',
      license='MIT',
      package_dir={'': 'src/'},
      packages=['gnpp'],
      zip_safe=False)