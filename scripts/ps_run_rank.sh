gradient jobs create \
    --projectId 'prc8cunsc' \
    --name 'Experiments (CV)' \
    --command 'HOST_TYPE=remote python3 main.py --variant shared --cv --experiment rank' \
    --useDockerfile true \
    --machineType C7
