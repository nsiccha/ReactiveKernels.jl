# Per-row metadata for the all-82 benchmark: `family`, the truthful structural-difference
# `note` (primal-table column), and the OPTIONAL-cell provenance (optimized-Stan /
# further-optimized-Turing). The optional optimized columns are DEFERRED by an EXPLICIT USER
# directive (2026-09-07): "No need to add any further hand optimized variants for this table
# — that's definitely for later." That user directive is the truthful SOURCE-PROVENANCE for
# their absence (NOT a grep/heuristic inference). If a distinct VERIFIED-FASTER optimized
# implementation is later wired, the corresponding cell becomes a real number instead.
module All80Metadata

const OPT_STAN_NA = "deferred (user directive 2026-09-07): optimized-Stan variants out of scope for this table; reference Stan is the Stan comparator"
const FURTHER_TURING_NA = "deferred (user directive 2026-09-07): further-optimized Turing variants out of scope for this table; upstream Turing is the Turing comparator"

# NOTES :: Dict(key => (family=..., note=...)) — filled from source (each .stan idiom vs the
# RK @kernel authoring). Present for every one of the 82 keys.
include(joinpath(@__DIR__, "all80_metadata_notes.jl"))

"""Metadata for one model key: family + structural-difference note + the four optional-cell
provenance strings. A key missing from NOTES yields empty family/note (a completeness gap
the caller can flag), never a fabricated note."""
function meta(key)
    fn = get(NOTES, key, (family = "", note = ""))
    (; family = String(fn.family), note = String(fn.note),
       primal_opt_stan = OPT_STAN_NA, primal_further_turing = FURTHER_TURING_NA,
       gradient_opt_stan = OPT_STAN_NA, gradient_further_turing = FURTHER_TURING_NA)
end

missing_keys(keys) = [k for k in keys if !haskey(NOTES, k) || isempty(get(NOTES, k, (note="",)).note)]

end # module All80Metadata
