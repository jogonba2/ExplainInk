#!/usr/bin/env bash

set -e
set -x

mypy "explainink"
flake8 "explainink" --ignore=E501,W503,E203,E402,E704
black "explainink" --check -l 80
