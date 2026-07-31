"""Define the typed representation of a single configurable model parameter."""

from dataclasses import asdict, dataclass


@dataclass
class ModelParam:
    """Dataclass for a model parameter, which compose the ModelConfig."""

    param_name: str
    value: float
    description: str
    opti_min: float
    opti_max: float

    @classmethod
    def from_dict(cls, dict_data):
        """Create a model parameter from a dictionary."""
        return cls(**dict_data)

    def asdict(self):
        """Return a dictionary representation of the model parameter."""
        return asdict(self)

    def as_config_dict(self):
        """Return the parameter in the JSON config file shape."""
        return {
            "value": self.value,
            "description": self.description,
            "opti_min": self.opti_min,
            "opti_max": self.opti_max,
        }
