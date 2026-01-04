import yaml


class Config:
    """Lightweight config wrapper supporting dotted keys (e.g. 'coil_design.wf_nPhi')."""

    def __init__(self, data: dict):
        self._data = data

    def get(self, key: str, default=None):
        if not isinstance(key, str):
            return default
        keys = key.split(".")
        val = self._data
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    def to_dict(self) -> dict:
        return self._data

    def to_string(self) -> str:
        return yaml.dump(self._data)
