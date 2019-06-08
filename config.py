import os

on_remote = False
if 'HOST_TYPE' in os.environ and os.environ['HOST_TYPE'] == 'remote':
    print('Running on remote machine!')
    on_remote = True


OUT_DIR = '/artifacts' if on_remote else './out'
VIS_DIR = '/artifacts/vis' if on_remote else './vis'


def prepare_env():
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    if not os.path.exists(VIS_DIR):
        os.mkdir(VIS_DIR)
