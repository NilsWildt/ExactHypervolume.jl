#!/usr/bin/env bash
# JuliaCheck — the advisory source pass.
#
# Needs the *working copy*, not the registered package: the curated rules this
# workflow relies on are local additions that upstream does not ship. Running
# `julia -m JuliaCheck` against a registered install fails loudly with
# "Unknown rules".
#
#     bash qa/juliacheck.sh
#
# JuliaCheck's exit status is 0 whether or not violations are found, so the
# findings are what matter, never the exit code. This is advisory: it never
# blocks. The gate is qa/run.jl.

set -uo pipefail

JULIACHECK="${JULIACHECK:-$HOME/JuliaCheck.jl}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$JULIACHECK" ]]; then
    echo "JuliaCheck working copy not found at $JULIACHECK" >&2
    echo "Set JULIACHECK=/path/to/JuliaCheck.jl and retry." >&2
    exit 2
fi

julia --project="$JULIACHECK" -m JuliaCheck --output simple --enable \
    struct-fields-have-concrete-types \
    concrete-structs-avoid-runtime-bool-fields \
    avoid-containers-with-abstract-types \
    avoid-try-catch-in-functions \
    variables-have-fixed-types \
    too-many-types-in-unions \
    implement-unions-as-consts \
    use-isnan-to-check-for-nan \
    use-isinf-to-check-for-infinite \
    use-ismissing-to-check-for-missing-values \
    use-isnothing-to-check-for-nothing-values \
    do-not-comment-out-code \
    exclamation-mark-in-function-identifier-if-mutating \
    -- "$HERE/src" "$HERE/test" "$HERE/qa" \
    | grep -v '^>> Processing file' \
    | grep -v '^$'

# Rules deliberately left off, and why:
#
#   use-eachindex-to-iterate-indices
#       The package calls `Base.require_one_based_indexing` on every argument it
#       receives and documents the 1-based contract; `test_api.jl` proves a
#       shifted axis is refused rather than misread. Internal buffers are plain
#       `Vector`/`Matrix`, so `1:n` is exact, and on a `d × n` matrix
#       `eachindex` would iterate elements rather than one axis anyway.
#
#   functions-mutate-only-zero-or-one-arguments
#       Every flagged function threads a pre-allocated workspace through the
#       sweeps on purpose. The argument is provably not modified: "public
#       functions leave their inputs untouched" in test_api.jl compares the
#       caller's matrix and reference point against a copy after every entry
#       point has run.
#
#   function-identifiers-in-lower-snake-case, type-names-upper-camel-case
#       These fire on type constructors (`BoxPoint`, `GaussianPoint`) and on
#       acronyms (`WFGWorkspace`), both of which are already correct.
