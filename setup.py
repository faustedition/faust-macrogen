from setuptools import setup, find_packages

setup(
        name='faust-macrogen',
        version='0.9',
        packages=find_packages('src'),
        package_dir={'': 'src'},
        url='',
        license='',
        author='Thorsten Vitt',
        author_email='thorsten.vitt@uni-wuerzburg.de',
        description='Macrogenesis Tools for the Faustedition',
        include_package_data=True,
        install_requires=[
                'networkx>=2.1',
                'python-igraph',
                'pygraphviz',
                'ruamel.yaml',
                'pandas',
                'openpyxl',
                'xlrd',
                'lxml',
                'requests',
                'requests-cache',
                'more-itertools',
                'logging-tree',
                'colorlog',
                'tqdm',
                'dataclasses',
                'cvxpy'
        ],
        entry_points={
            'console_scripts': ['macrogen=macrogen.main:main']
        }
)
