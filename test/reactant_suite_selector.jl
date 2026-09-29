# Pure group/file selector for the optional Reactant compiler suite.
#
# `test_reactant.jl` is one entrypoint over nine suites: eight separately
# authored `*_reactant.jl` files plus the inline `core` compiler-integration
# testset. The whole entrypoint is too heavy for one process on a squeezed
# fleet host, so the entrypoint accepts an `ARGS` selector:
#
#   julia --project=<reactant-env> test/test_reactant.jl [selector...]
#
# Each selector names a group (`core`, `pathfinder`, ...) or a suite file
# (`test_pathfinder_reactant.jl`, with or without the `.jl`). With no
# selector every suite runs, which is the historical full behavior that CI's
# `RK_REACTANT_TESTSET=all` path relies on. On memory-squeezed fleet hosts
# run one group per process instead of the whole file.
#
# This file is dependency-free (no Reactant) so the core suite can
# regression-test the selection logic itself; both `test_reactant.jl` and
# `test/runtests.jl` include it at top level.

const _REACTANT_SUITE_ORDER = (
    "plate-chains",
    "ref-array-plate",
    "core",
    "nutpie",
    "reactivehmc-statistics",
    "reactivehmc-hmc",
    "finite-structural-container",
    "kernel-nuts",
    "pathfinder",
)

# Suite file per group in `_REACTANT_SUITE_ORDER` order; `nothing` marks the
# inline `core` testset, which has no file of its own.
const _REACTANT_SUITE_FILES = (
    "test_authored_plate_chains_reactant.jl",
    "test_ref_array_plate_reactant.jl",
    nothing,
    "test_nutpie_reactant.jl",
    "test_reactivehmc_statistics_reactant.jl",
    "test_reactivehmc_hmc_reactant.jl",
    "test_finite_structural_container_reactant.jl",
    "test_kernel_nuts_reactant.jl",
    "test_pathfinder_reactant.jl",
)

function _reactant_suite_group_files(selector::String)
    selector in _REACTANT_SUITE_ORDER && return (selector,)
    nothing
end

function _reactant_suite_file_selector(selector::String)
    basename(selector) == selector || return nothing
    file = endswith(selector, ".jl") ? selector : selector * ".jl"
    for (group, group_file) in zip(_REACTANT_SUITE_ORDER, _REACTANT_SUITE_FILES)
        group_file == file && return group
    end
    nothing
end

function _select_reactant_suites(selectors::AbstractVector{<:AbstractString})
    isempty(selectors) && return _REACTANT_SUITE_ORDER

    requested = Set{String}()
    unknown = String[]
    for raw_selector in selectors
        selector = String(raw_selector)
        group = _reactant_suite_group_files(selector)
        if group !== nothing
            union!(requested, group)
            continue
        end
        suite = _reactant_suite_file_selector(selector)
        if suite === nothing
            push!(unknown, selector)
        else
            push!(requested, suite)
        end
    end

    if !isempty(unknown)
        unknown = sort!(unique!(unknown))
        known_files = filter(!isnothing, _REACTANT_SUITE_FILES)
        throw(ArgumentError(
            "unknown Reactant suite selector(s): $(join(repr.(unknown), ", ")). " *
            "Known groups: $(join(_REACTANT_SUITE_ORDER, ", ")). " *
            "Known files: $(join(known_files, ", "))"))
    end

    Tuple(suite for suite in _REACTANT_SUITE_ORDER if suite in requested)
end

@assert length(_REACTANT_SUITE_ORDER) == length(_REACTANT_SUITE_FILES)
@assert _select_reactant_suites(String[]) == _REACTANT_SUITE_ORDER
@assert _select_reactant_suites(collect(_REACTANT_SUITE_ORDER)) ==
    _REACTANT_SUITE_ORDER
