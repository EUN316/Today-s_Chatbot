import yaml

def load_config(path="configs/Base.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return cfg: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
