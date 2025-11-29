from common.constants import SERVICE_CONF
import os
from common.file_utils import get_project_base_directory
from ruamel.yaml import YAML

def load_yaml_conf(conf_path):
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(conf_path) as f:
            yaml = YAML(typ="safe", pure=True)
            return yaml.load(f)
    except Exception as e:
        raise EnvironmentError("loading yaml file config from {} failed:".format(conf_path), e)

def conf_realpath(conf_name):
    conf_path = f"conf/{conf_name}"
    return os.path.join(get_project_base_directory(), conf_path)

def read_config(conf_name=SERVICE_CONF):
    local_config = {}
    local_path = conf_realpath(f'local.{conf_name}')

    # load local config file
    if os.path.exists(local_path):
        local_config = load_yaml_conf(local_path)
        if not isinstance(local_config, dict):
            raise ValueError(f'Invalid config file: "{local_path}".')

    global_config_path = conf_realpath(conf_name)
    global_config = load_yaml_conf(global_config_path)

    if not isinstance(global_config, dict):
        raise ValueError(f'Invalid config file: "{global_config_path}".')

    global_config.update(local_config)
    return global_config

CONFIGS = read_config()

def get_base_config(key, default=None):
    if key is None:
        return None
    if default is None:
        default = os.environ.get(key.upper())
    return CONFIGS.get(key, default)
