#!/usr/bin/env python3
"""Run only pinned IP102 bounded metadata inspection; no archive download."""
from probe_zip_layout import URLS, inspect

if __name__ == '__main__':
    inspect('ip102', URLS['ip102'])
