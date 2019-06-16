import os


OUT_DIR = os.path.join(get_results_dir(), 'out')
VIS_DIR = os.path.join(get_results_dir(), 'vis')
DATA_DIR = get_storage_dir()


def on_remote():
    """
    You should set `HOST_TYPE` env variable if on remote.
    Look at the `scripts /` directory
    """
    key = 'HOST_TYPE'
    return key in os.environ and os.environ[key] == 'remote'


def get_results_dir():
    if on_remote():
        return '/artifacts'

    return './'


def get_storage_dir():
    if on_remote():
        return '/storage/EnsembleDiversityData'

    return '../EnsembleDiversityData'


def prepare_env():
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    if not os.path.exists(VIS_DIR):
        os.mkdir(VIS_DIR)
