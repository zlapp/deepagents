"""Middleware for the DeepAgent."""

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

__all__ = [
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "SkillsMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
]


def __getattr__(name: str):
    """Lazy import for SkillsMiddleware to avoid circular import issues."""
    if name == "SkillsMiddleware":
        try:
            from deepagents_cli.skills.middleware import SkillsMiddleware
            return SkillsMiddleware
        except ImportError:
            raise ImportError(
                "SkillsMiddleware requires deepagents-cli to be installed. "
                "Install it with: pip install deepagents-cli"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
