"""Thin per-shell adapters, shipped as data files and installed by `cl-ai init`.

Each holds exactly one piece of state: which candidate is highlighted. All
intelligence lives in the daemon, so each adapter is written once and rarely
touched. This is deliberate -- it is the layer there are four of.
"""
