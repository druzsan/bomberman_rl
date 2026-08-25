"""Shared, dependency-light primitives for every agent in this repository.

The package is vendored into each agent directory by ``tools/sync_lib.py`` so a
submitted agent stays self-contained.  Nothing in here may import repo-root
modules (``settings``, ``environment``, ...): the game constants are duplicated
in :mod:`lib.board` on purpose, because training monkeypatches ``settings`` and
the shipped agent must never depend on a patched value.
"""
