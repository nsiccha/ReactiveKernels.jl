# Standalone reproduction of the Survey Reactant-AD status on f1e8b83.
#
# The reactant phase historically pre-declared `Survey_data-Survey_model` in
# `all80_reactant_body.jl`'s `PROCESS_ABORTING_AD` because its Enzyme AD terminated the whole
# subprocess with signal 6 (uncatchable) — a 1-D discrete-count marginalization whose
# generated-call/GC marking aborted. This script re-attempts that AD IN ISOLATION: the exact
# `run_reactant_one` setup, the real `prepare_ad` + gradient `@compile` + a finite-gradient
# evaluation, with NO skip. A SIGABRT kills only this throwaway process (wrap in a subshell and
# read the exit code: 134 == 128+6 == signal 6). On 2026-09-08 (HEAD c75242b) it SURVIVES and
# prints SURVEY_AD_OK, which is why `PROCESS_ABORTING_AD` is now empty and Survey is measured
# like every other model. Re-run this if a future change is suspected to reintroduce the abort.
#
# Run: julia --startup-file=no --project=benchmark/all80-env benchmark/survey_ad_probe.jl
using Random, LinearAlgebra, Statistics
import BridgeStan, PosteriorDB, Reactant, Enzyme
using ReactiveKernels, ReactiveKernelsPPLExamples
using DifferentiationInterface

include(joinpath(@__DIR__, "all80_registry.jl"))
const RK = All80Registry.REGISTRY
const AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

name = "Survey_data-Survey_model"
println("=== SURVEY_AD_PROBE start (isolated) ==="); flush(stdout)
post = PosteriorDB.posterior(PosteriorDB.database(), name)
data = PosteriorDB.load(PosteriorDB.dataset(post))
entry = RK[name]
graph = getproperty(getproperty(ReactiveKernelsPPLExamples, entry.mod), entry.build)()
kb = prepare(graph; have = entry.have, want = :posterior, bound = entry.bind(data))

stan = PosteriorDB.implementation(PosteriorDB.model(post), "stan")
library = first(splitext(PosteriorDB.path(stan))) * "_model.so"
sm = BridgeStan.StanModel(library, PosteriorDB.load(PosteriorDB.dataset(post), String), 468)
dim = Int(BridgeStan.param_unc_num(sm))

function find_point(kb, dim, entry)
    okq(x) = length(x) == dim && isfinite(try kb(x) catch; NaN end)
    entry.probe_q !== nothing && okq(entry.probe_q) && return collect(Float64, entry.probe_q)
    rng = Xoshiro(0xC0FFEE)
    for cand in Iterators.flatten(([fill(0.1, dim), fill(-0.1, dim)],
                                   (0.5 .* randn(rng, dim) for _ in 1:96), (zeros(dim),)))
        okq(cand) && return collect(Float64, cand)
    end
    error("no valid probe point")
end
q = find_point(kb, dim, entry)
println("dim=$dim  primal kb(q)=", kb(q)); flush(stdout)

println(">>> prepare_ad(kb, AE, q; active=:unconstrained) — abort candidate #1"); flush(stdout)
prep = prepare_ad(kb, AE, q; active = :unconstrained)
println("prepare_ad SURVIVED"); flush(stdout)

println(">>> gradient @compile ad_value_and_gradient! — abort candidate #2 (Enzyme generated-call)"); flush(stdout)
rq = Reactant.to_rarray(q)
gb = Reactant.to_rarray(similar(q))
gradc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep, gb, rq)
println("gradient @compile SURVIVED"); flush(stdout)
_, rgrad = gradc(prep, gb, rq)
ghost = Array{Float64}(rgrad)
println("gradient EVALUATED; finite=", all(isfinite, ghost),
        " first=", first(ghost, min(3, length(ghost)))); flush(stdout)
# A non-abort is only "OK" if the gradient is actually finite — otherwise SURVEY_AD_OK would lie
# about a NaN/Inf result (the production reactant_cells enforces finite + Stan parity; this keeps
# the standalone probe from regressing to a false pass).
all(isfinite, ghost) || error("Survey gradient is NON-FINITE: $(ghost) — probe is NOT ok")
println("SURVEY_AD_OK — finite gradient, the historical process-abort no longer reproduces on this base")
