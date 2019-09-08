#!/bin/sh

gradient jobs create \
    --projectId 'prc8cunsc' \
    --name 'Experiments (CV)' \
    --command 'HOST_TYPE=remote python3 main.py --variant shared --cv 5 --experiment diversity' \
    --useDockerfile true \
    --machineType C7
