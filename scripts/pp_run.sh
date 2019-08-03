gradient jobs create \
    --projectId 'prc8cunsc' \
    --name 'Experiments' \
    --command 'HOST_TYPE=remote python3 main.py --variant shared' \
    --useDockerfile true \
    --machineType C7
