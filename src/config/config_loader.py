import yaml


def load_config(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)
