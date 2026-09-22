"""Assembles ContextFacts. Runs on every Tab, so everything is cached with cheap
invalidation (PATH scan until PATH changes; directory markers keyed on mtime).
"""
