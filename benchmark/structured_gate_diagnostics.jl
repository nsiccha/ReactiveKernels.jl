#!/usr/bin/env julia

# structured_gate_diagnostics.jl — OPT-IN experiments for the documented
# static-activity limitation (snag scan-prior-enzym-d67d4ac1). This is NOT a
# correctness gate and NOT part of package acceptance. HISTORICAL EVIDENCE
# PRODUCER: per the user's 2026-09-09 direction ("I generally forbid that
# anything rk requires runtime_activity to work with enzyme") these probes are
# diagnostic evidence only — they are NOT rerun for this delivery, are not
# accepted support, and no HMM delivery may require them or any priming through
# this mode. The native plain-Reverse failure remains an unresolved RK defect
# under snag scan-prior-enzym-d67d4ac1. It records TWO SEPARATE facts: (a) the
# ordinary vs Const-annotated native gradient configuration comparison; and
# (b) the runtime-activity gradient against Stan (proof the translation's
# gradient math is correct). See snag scan-prior-enzym-d67d4ac1.
#
# Run: julia --project=benchmark/all80-env benchmark/structured_gate_diagnostics.jl

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

const PE = ReactiveKernelsPPLExamples
const AE_ORD = AutoEnzyme(mode = Enzyme.Reverse)
const AE_CONST = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
const AE_RTA = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                          function_annotation = Enzyme.Const)
sval(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
sgrad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_mat(x) = x isa AbstractMatrix ? Float64.(x) :
          reduce(vcat, [permutedims(Float64.(r)) for r in x])

for (name, build, have, bind) in (
    ("bball_drive_event_1-hmm_drive_1", () -> PE.HmmDrive1Example.build_hmm_drive_1_graph(),
     (:unconstrained, :u, :v, :alpha, :tau, :rho),
     d -> (u = Float64.(d["u"]), v = Float64.(d["v"]), alpha = _mat(d["alpha"]),
           tau = Float64(d["tau"]), rho = Float64(d["rho"]))),
    ("bball_drive_event_0-hmm_drive_0", () -> PE.HmmDrive0Example.build_hmm_drive_0_graph(),
     (:unconstrained, :u, :v, :alpha),
     d -> (u = Float64.(d["u"]), v = Float64.(d["v"]), alpha = _mat(d["alpha"]))))

    println("\n########## diagnostics: $name ##########"); flush(stdout)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    sp = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan"))
    sm = BridgeStan.StanModel(sp, PosteriorDB.load(PosteriorDB.dataset(post), String), 468)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    kb = prepare(build(); have, want = :posterior, bound = bind(data))
    rng = Xoshiro(468)
    q = 0.5 .* randn(rng, Int(BridgeStan.param_unc_num(sm)))
    @assert isfinite(sval(sm, q)) && all(isfinite, sgrad(sm, q))

    for (label, AE) in (("ordinary Reverse", AE_ORD), ("Const-annotated", AE_CONST))
        err = try
            prep = prepare_ad(kb, AE, q; active = :unconstrained)
            ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)
            nothing
        catch e
            sprint(showerror, e, catch_backtrace())
        end
        println("  native $label : ", err === nothing ? "OK" : "documented failure: " * first(split(err, '\n'))); flush(stdout)
    end

    prep_rta = prepare_ad(kb, AE_RTA, q; active = :unconstrained)
    g = ReactiveKernels.ad_value_and_gradient!(prep_rta, similar(q), q)[2]
    rg = relerr(g, sgrad(sm, q))
    @assert all(isfinite, g) && rg < 1e-3 "$name: runtime-activity gradient rel=$rg"
    println("  runtime-activity grad vs Stan rel=$(round(rg; sigdigits=4)) PASS (math diagnostic)"); flush(stdout)
end
println("\ndiagnostics done"); flush(stdout)
