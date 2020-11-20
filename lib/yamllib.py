import yaml

def read_yaml(yaml_file):
  with open(yaml_file, "r") as fh:
    python_object = yaml.load(fh, Loader=yaml.SafeLoader)
    return python_object