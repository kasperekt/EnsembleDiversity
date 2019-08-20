gradient jobs create \
    --projectId 'prc8cunsc' \
    --name 'Validate structures' \
    --command 'HOST_TYPE=remote python3 validate_structure.py -A --verbose' \
    --useDockerfile true \
    --machineType C7
