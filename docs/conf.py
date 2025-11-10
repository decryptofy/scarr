import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'SCARR'
author = 'decryptofy'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
