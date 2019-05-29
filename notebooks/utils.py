import os
import sys


def prepare_jupyter():
    cwd = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if cwd not in sys.path:
        sys.path.append(cwd)
