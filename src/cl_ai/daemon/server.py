"""Resident process, single instance per user. Owns the warm model, embedding
index, catalog, and the agent LRU.

Measured justification: per-process invocation costs 579ms before answering
anything; warm in-process is 9.7ms to embed and 68ms to complete. Tab has to
land under ~100ms, so a daemon is mandatory, not an optimisation.

Every request carries a deadline and returns best-effort partial results rather
than overrunning it.

Open discovery on a new machine harvests in the BACKGROUND: the daemon serves
whatever catalog it already has and hot-swaps the index when a batch lands. The
user never waits on harvesting.
"""
