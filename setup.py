from distutils.core import setup
from pathlib import Path

def get_install_requires():
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets

requirements = get_install_requires()

# The version is part of OmniChem's results-cache key and is printed in every exported result: raise it whenever the code in this folder or the
# checkpoints it is used with change (1.1.0: the OmniChem fork - transformers 4.52.4, timm shim; upstream is 1.0).
setup(name='RxnScribe',
      version='1.1.0',
      description='RxnScribe',
      author='Yujie Qian',
      author_email='yujieq@csail.mit.edu',
      url='https://github.com/Ozymandias314/MolDetect',
      packages=['rxnscribe', 'rxnscribe.inference', 'rxnscribe.pix2seq', 'rxnscribe.transformer'],
      package_dir={'rxnscribe': 'rxnscribe'},
      package_data={},
      setup_requires=['numpy'],
      install_requires=requirements,
      )