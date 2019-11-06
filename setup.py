# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='ChineseNER',
    version='0.1',

    description='基于BiLSTM-CRF的字级别中文命名实体识别模型',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/cswangjiawei/ChineseNER',

    # Author details
    author='wangjiawei',
    author_email='cswangjiawei@163.com',

    # Choose your license
    license='MIT',

    # What does your project relate to?
    keywords='Named-entity recognition using neural networks',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    zip_safe=False,
    package_data={
    'ChineseNER': ['data/*', 'model/*'],
}, 
	install_requires=['numpy', 'torch', 'tensorboardX']
)