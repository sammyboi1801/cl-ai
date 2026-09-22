"""Needle 3 backend.

Owns every Needle-specific quirk so core never sees it:
  - LRU agent cache keyed by toolset hash (measured: 270ms to construct, 68ms
    to reuse; retrieval hands us a different toolset each query)
  - embedding centering
  - refusal normalisation (empty function_calls == refusal, no free text)
  - confidence is nullable (local finetune does not train the calibration head)

Both `confidence` and `reasoning` are captured as debug-only. Observed twice:
the reasoning string describes a different tool than the one actually called.
"""
