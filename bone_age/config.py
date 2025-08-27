from pathlib import Path
import yaml

def load_config(path: str | None = None):
    p = Path(path or "config/defaults.yaml")
    with open(p, 'r') as file:
        return yaml.safe_load(file)