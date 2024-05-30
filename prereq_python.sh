#!/bin/bash

# Upgrade pip
python -m pip install --upgrade pip

# Force install six and appdirs to avoid issues.
python -m pip install --ignore-installed six appdirs
