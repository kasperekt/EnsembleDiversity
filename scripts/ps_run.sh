#!/bin/sh

gradient jobs create \
    --projectId 'prc8cunsc' \
    --name 'Experiments' \
    --command 'HOST_TYPE=remote python3 main.py --variant shared --experiment diversity --cv 10 --reps 10' \
    --useDockerfile true \
    --machineType C7
