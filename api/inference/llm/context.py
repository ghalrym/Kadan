"""Checkpoint context limits shared by native language adapters."""
class ContextLimitError(ValueError):
    pass


class ContextMemoryError(RuntimeError):
    pass


def _get(config, key, default=None):
    """Read a configuration key from either a mapping or an attribute-based object."""
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def text_config(config):
    """Return the nested text configuration when present, otherwise the supplied configuration."""
    return _get(config, 'text_config', None) or config


def resolve_context(config, configured=None):
    """Return (architecture maximum, effective limit). None selects the maximum; invalid or
    excessive limits raise ContextLimitError without truncation.
    """
    supported = _get(text_config(config), 'max_position_embeddings')
    if isinstance(supported, bool) or not isinstance(supported, int) or supported < 1:
        raise ContextLimitError('Checkpoint does not declare a positive architecture context limit')
    if configured is not None and (isinstance(configured, bool) or not isinstance(configured, int) or not 1 <= configured <= supported):
        raise ContextLimitError(f'Configured context must be between 1 and {supported} tokens, or null for the architecture maximum')
    return supported, supported if configured is None else configured
