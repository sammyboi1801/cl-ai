"""Per-shell recent history: PSReadLine ConsoleHost_history.txt, .bash_history,
zsh HISTFILE, fish. Recency-weighted stack boost decaying over minutes, not
commands.

Stays local -- feeds scoring only, never leaves the machine.
"""
