import os

OUT_DIR = './out'
VIS_DIR = './vis'


def prepare_env():
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    if not os.path.exists(VIS_DIR):
        os.mkdir(VIS_DIR)
