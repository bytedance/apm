import setuptools as tools

tools.setup(
    name="apm",
    packages=[
        'openfold',
        'apm',
        'ProteinMPNN'
    ],
    package_dir={
        'openfold': './openfold',
        'apm': './apm',
        'ProteinMPNN': './ProteinMPNN',
    },
)
