"""Schema validation, type coercion, enum snapping, required-field checks.

A no-op under Needle's grammar guarantee. It exists so that guarantee is never
load-bearing -- core must program against the weakest backend.
"""
