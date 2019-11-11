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
        setup_requires=[
            'numpy'       # work around https://github.com/numpy/numpy/issues/2434
        ],
        install_requires=[
                'numpy',
                'more-itertools',
                'networkx>=2.1,<2.4',
                'python-igraph',
                'pygraphviz',
                'ruamel.yaml',
                'pyyaml',    # required for networkx' yaml dump
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
                'cvxpy',
                'cvxopt'   # just used to pull in GLPK
        ],
        entry_points={
            'console_scripts': ['macrogen=macrogen.main:main']
        }
)
