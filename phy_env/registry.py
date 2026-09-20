"""Component registry for policy, observation, and scoring components."""

COMPONENT_REGISTRY = {}


def register_component(category, name):
    """Register a component class under a category and name."""

    def wrapper(cls):
        COMPONENT_REGISTRY.setdefault(category, {})[name] = cls
        return cls

    return wrapper


def build_component(category, cfg):
    """Instantiate a registered component from cfg.type."""
    name = cfg.type
    if category not in COMPONENT_REGISTRY or name not in COMPONENT_REGISTRY[category]:
        raise ValueError(f"unregistered component: category={category}, type={name}")
    return COMPONENT_REGISTRY[category][name](cfg)
