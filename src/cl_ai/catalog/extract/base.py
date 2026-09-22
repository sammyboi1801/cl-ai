"""Stage 2 -- source-tier extractors. Each emits raw findings with provenance.

Source protocol: tier, availability probe, extract(tool) -> RawFindings.

Tier ladder (stop at the first hit):
  0 native introspection   pwsh Get-Command, argparse, clap/cobra metadata
  1 machine-readable help  --help=json, doc emitters
  2 completion scripts     bash/zsh/fish -- often carry valid VALUE SETS
  3 man + local HTML docs  e.g. Git's 510 bundled git-doc/*.html
  4 tldr placeholders      argument semantics + examples
  5 --help scraping        opt-in only; requires --exec

Every harvested field carries its tier and a confidence score, so the linter
can say "this enum came from tier 5, do not trust it".
"""
