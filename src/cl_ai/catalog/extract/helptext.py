"""Tier 5. --help scraping. OPT-IN (--exec), never the default.

Traps, all observed on this machine with git:
  - `git commit --help` prints NOTHING -- it opens a browser
  - `git commit -h` writes to STDERR and exits 129
So: capture both streams, ignore the exit code, try several flag conventions.

Safety, because this executes arbitrary binaries found on PATH: process-tree
kill on timeout, stdin from null, PAGER=cat GIT_PAGER=cat NO_COLOR=1 TERM=dumb,
output size cap, temp cwd, and refuse PATH entries writable by others.
"""
