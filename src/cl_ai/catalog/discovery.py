"""Stage 1 -- inventory. What binaries/builtins/functions exist here.

Open discovery, exec-free: PATH scan (PATHEXT aware), shell builtins, aliases,
package-manager manifests. Emits inventory.json.

A tool discovered WITHOUT a schema is a first-class result, not a failure: it
lets us say "I know git clone exists but cannot fill it in" instead of
substituting a neighbour.
"""
