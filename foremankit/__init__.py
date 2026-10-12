"""foremankit: delegate Claude Code work to cheaper / local models."""
import os
from collections.abc import MutableMapping

__version__ = "0.1.0"

LEGACY_ENV_PREFIX = "AGENTKIT_"
ENV_PREFIX = "FOREMANKIT_"


def adopt_legacy_env(environ: MutableMapping[str, str]) -> list[str]:
    """Copy variables of the old name (``AGENTKIT_*``) to the new one (``FOREMANKIT_*``).

    The project was renamed; an existing machine may still export the old names. A variable that is
    already set under the new name wins and is never overwritten. Returns the new names it created.
    """
    created = []
    for name in sorted(environ):
        new_name = ENV_PREFIX + name[len(LEGACY_ENV_PREFIX):]
        if name.startswith(LEGACY_ENV_PREFIX) and new_name not in environ:
            environ[new_name] = environ[name]
            created.append(new_name)
    return created


adopt_legacy_env(os.environ)
