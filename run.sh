#!/bin/bash

PYTHON_BIN=python3

nohup $PYTHON_BIN -u main.py > experiments.txt &
