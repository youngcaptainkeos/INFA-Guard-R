"""
Defender model implementations

This package is part of the MAS Collusion Guard project.
"""

__version__ = "1.0.0"
__author__ = "MAS Collusion Guard Team"

# Import key modules for easier access
from .gat_with_attr_conv import *
from .model import *

__all__ = ['gat_with_attr_conv', 'model']
