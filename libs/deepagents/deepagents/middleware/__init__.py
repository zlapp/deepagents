"""Middleware for the DeepAgent."""

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

try:
    from deepagents_cli.skills.middleware import SkillsMiddleware
except ImportError:
    SkillsMiddleware = None  # type: ignore[assignment, misc]

__all__ = [
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "SkillsMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
]
