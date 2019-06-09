import os


def get_results_dir():
    key = 'HOST_TYPE'

    if key in os.environ and os.environ[key] == 'remote':
        return '/artifacts'

    return './'


OUT_DIR = os.path.join(get_results_dir(), 'out')
VIS_DIR = os.path.join(get_results_dir(), 'vis')


def prepare_env():
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    if not os.path.exists(VIS_DIR):
        os.mkdir(VIS_DIR)
