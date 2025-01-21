#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "explainink" "examples" --exclude=__init__.py
isort "explainink" "examples"
black "explainink" "examples" -l 80
