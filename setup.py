import setuptools

with open('README.md','r') as file:
    long_description = file.read()

setuptools.setup(
    name = 'preprocess_text', #should be unique globally to avoid package name conflict
    version = '1.6.5',
    author = 'Hwai Teng Teoh',
    author_email = 'teoh0821@gmail.com',
    description = 'This is text preprocessing package',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages = setuptools.find_packages(),
    classifiers = [
    'Programming Language :: Python :: 3',
    'Operating System :: OS Independent'],
    python_requires = '>=3.5'
)