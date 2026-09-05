# Shared reproducibility guard for the pinned external-environment benchmark
# scripts: the candidate MUST be a tracked-clean, DETACHED-HEAD worktree at the
# exact committed SHA, so an expensive receipt can never be produced from an
# attached branch (whose ref could move, or hold uncommitted/held WIP) — only from
# an immutable pinned checkout.
function _require_clean_detached_candidate(root)
    sha = readchomp(`git -C $root rev-parse HEAD`)
    abbrev = readchomp(`git -C $root rev-parse --abbrev-ref HEAD`)
    abbrev == "HEAD" || error(
        "reproducible benchmark run requires a DETACHED-HEAD worktree at the exact " *
        "committed SHA, but HEAD is attached to branch `$(abbrev)`. Create one with " *
        "`git worktree add --detach <path> <sha>` and run from there.")
    dirty = readchomp(`git -C $root status --porcelain --untracked-files=no`)
    isempty(dirty) || error(
        "reproducible benchmark run requires a tracked-clean candidate; commit or " *
        "revert these changes first:\n$(dirty)")
    sha
end

# Keep pinned-environment setup separate from the explicit Pkg.precompile()
# receipt. Julia 1.10's parallel precompile can deadlock while sibling Reactant
# extensions wait on locks held by their own driver; one task avoids that cycle.
# Scope both settings to setup so the benchmark child keeps the caller's env.
function _with_serial_pkg_precompile(f)
    withenv(f, "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            "JULIA_NUM_PRECOMPILE_TASKS" => "1")
end
