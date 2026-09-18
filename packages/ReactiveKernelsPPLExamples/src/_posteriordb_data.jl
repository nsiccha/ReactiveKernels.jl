# Shared PosteriorDB data loader for the faithful posteriordb example modules.
#
# Each Example module sources its REAL, FULL data from the bundled posteriordb-1.0.0
# artifact via `PosteriorDB.jl` (v0.6.x) instead of hand-inlined arrays. `_posteriordb_data`
# opens the database once and caches each posterior's loaded data Dict, so all ~90 modules
# calling it at module TOP LEVEL (during precompilation) share one open database and never
# re-parse a dataset.
#
# The data are keyed by the upstream `.stan` data-block variable names (e.g. "y", "sigma",
# "J", "year", "C"). Each module maps those keys to its bound ports at load time, applying
# the numeric conversions the graph expects (`Float64.`/`Int.`) — the same mapping the
# benchmark registry uses, so nothing is copied inline and the full data is authoritative.

const _PDB_DATABASE = Ref{Any}()
const _PDB_CACHE = Dict{String,Any}()

"""
    _posteriordb_data(posterior_name) -> AbstractDict

Load (and cache) the real data Dict for a posteriordb posterior by name, e.g.
`_posteriordb_data("eight_schools-eight_schools_centered")`. Call from a module's
TOP LEVEL, so the loaded data is baked into the module's precompile cache — the
intended contract: downstream modules also consume these bindings at top level
(e.g. cross-module `using` of a data constant), so an `__init__`-time load would
leave them undefined during precompilation. The cached `Ref`/`Dict` themselves
are precompile-safe. Consequence of capturing at precompile time: if the bundled
posteriordb artifact ever changes or relocates without a `.jl` source change,
Julia will NOT re-precompile automatically — touch a source file or run
`Pkg.precompile` to refresh the baked data.
"""
function _posteriordb_data(posterior_name::AbstractString)
    get!(_PDB_CACHE, String(posterior_name)) do
        isassigned(_PDB_DATABASE) || (_PDB_DATABASE[] = PosteriorDB.database())
        post = PosteriorDB.posterior(_PDB_DATABASE[], posterior_name)
        PosteriorDB.load(PosteriorDB.dataset(post))
    end
end
