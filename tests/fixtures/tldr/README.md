# tldr page fixtures

Unmodified pages from [tldr-pages](https://github.com/tldr-pages/tldr),
copyright © 2014—present the tldr-pages team and contributors, licensed under
the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
(CC-BY-4.0).

They are vendored verbatim rather than paraphrased because their value as test
data is precisely that they are authentic. Each one pins a parsing edge case
found by scanning the full 7,367-page corpus:

| Page | What it pins |
| --- | --- |
| `common/git-commit.md` | The ordinary case: title split, `{{[-m\|--message]}}` alternation, a literal `--amend` flag, and a `More information:` homepage line |
| `common/aws-dynamodb.md` | Internally inconsistent brace escaping that cannot be parsed under any single rule; examples must be dropped with a warning, not guessed at |
| `common/acme.sh-dns.md` | A binary containing a dot whose subcommand is spelled as a flag (`acme.sh --dns`) |
| `common/rm.md`, `common/ls.md` | Destructive vs non-destructive capability inference |
| `linux/apt-get.md` | A hyphen that is part of the binary name, not a subcommand separator |
| `linux/pacman.md`, `pacman-sync.md`, `pacman-query.md` | Subcommands spelled as flags, which must stay distinct tools rather than collapsing into one bare `pacman` |
| `android/pm-install-commit.md` | A subcommand that itself contains a hyphen |
| `dos/mount.md` | Uppercase title, and `{{A:\}}` — a placeholder ending in a backslash, which is not an escape |
| `cisco-ios/dir.md` | An appliance CLI that must map to no host shell |

The full corpus is not committed: it is ~7k files, it lives outside the
installable package, and it changes upstream. `test_tldr_extract.py` checks it
additionally when `CL_AI_TLDR_ROOT` points at a local checkout. These fixtures
exist so the same edge cases are still covered on CI, where that corpus is
absent — a test that skipped there would report a pass while checking nothing.
