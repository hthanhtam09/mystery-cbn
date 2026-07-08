"""Concrete pipeline stages, grouped by processing domain (ARCHITECTURE.md §1.2, §15).

Stages register themselves; the kernel discovers them via the registry.
No stage imports another stage.
"""
