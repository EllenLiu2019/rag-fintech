import re
from common.constants import SERVICE_CONF
import os
from common import file_utils
from ruamel.yaml import YAML

# ${VAR:-default}
SETTING_PATTERN = r"\$\{(\w+)(?::(-[^}]*))?\}"


def load_yaml_conf(conf_path):
    if not os.path.isabs(conf_path):
        conf_path = file_utils.get_project_root_dir(conf_path)
    try:
        with open(conf_path) as f:
            yaml = YAML(typ="safe", pure=True)
            return yaml.load(f)
    except Exception as e:
        raise EnvironmentError("loading yaml file config from {} failed:".format(conf_path), e)


def read_config(conf_name=SERVICE_CONF):
    config_path = file_utils.get_project_root_dir("conf", conf_name)
    config_data = load_yaml_conf(config_path)

    if not isinstance(config_data, dict):
        raise ValueError(f'Invalid config file: "{config_path}".')

    return config_data


def get_base_config(key, default=None):
    if key is None:
        return None
    if default is None:
        default = os.environ.get(key.upper())
    config_data = read_config(SERVICE_CONF)
    return replace_env_vars(config_data.get(key, default))


def replace_env_vars(config):
    """Recursively replace environment variables in the configuration"""
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        return re.sub(
            pattern=SETTING_PATTERN,
            repl=lambda m: os.getenv(m.group(1), m.group(2)[1:] if m.group(2) else ""),
            string=config,
        )
    return config
