#!/usr/bin/env python
"""Read current version."""
import sys
import os.path as op


def main():
    sys.path.insert(0, op.abspath('.'))
    from nirodents.__about__ import __version__
    print(__version__)


if __name__ == '__main__':
    main()
