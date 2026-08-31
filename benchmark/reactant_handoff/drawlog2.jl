using ReactiveKernels, LinearAlgebra, Random
import ReactiveKernelsNUTSExamples
const RK = ReactiveKernels
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "examples", "nuts_runtime.jl"))
module Fix; include(joinpath(@__DIR__, "..", "nuts_kernel_authoring_fixture.jl")); end
# Define recorder INSIDE Random so the parentmodule domain gate admits it (no @generated world-age issue)
Random.eval(quote
    mutable struct RecRNG <: AbstractRNG
        inner::Xoshiro
        log::Vector{Any}
    end
end)
const RecRNG = Random.RecRNG
mkrec(seed) = RecRNG(Random.Xoshiro(seed), Any[])
Random.randn!(r::RecRNG, x::AbstractArray) = (Random.randn!(r.inner, x); push!(r.log,(:randn_momentum, copy(x))); x)
Base.rand(r::RecRNG, ::Type{Bool}) = (v=Base.rand(r.inner, Bool); push!(r.log,(:rand_bool_direction, v)); v)
Random.randexp(r::RecRNG) = (v=Random.randexp(r.inner); push!(r.log,(:randexp_multinomial, v)); v)

pf = RK._prepare_factory(Fix.euclidean_phasepoint, RK.kernel_registration(Fix.leapfrog!))
function vals(pf,T;metric=T[2 0;0 2])
    P=RK.kernel_prepared_plan(pf); d=Dict{Int,Any}()
    for s in RK.kernel_plan_slots(P)
        n=String(s.path[1]); d[s.canon] = n=="pot_f" ? (p->sum(abs2,p)) :
          n=="grad_f" ? ((dst,p)->(dst .= 2 .* p; sum(abs2,p))) :
          n=="metric" ? metric : n=="chol_metric" ? cholesky(metric) : startswith(n,"##node") ? zero(T) :
          n=="pos" ? T[1,2] : n=="mom" ? T[3,4] :
          n in ("dpot_dpos","dham_dpos","dkin_dmom","dham_dmom") ? T[0,0] : zero(T)
    end; d
end
function frame(pf,T,md;metric=T[2 0;0 2])
    f=ReactiveKernelsNUTSExamples._construct_nuts_frame(pf,vals(pf,T;metric),md;step_f=RK.partial(Fix.leapfrog!;stepsize=T(.1)),stats_f=Fix.nuts_stats!,min_dham=-1000.0)
    RK.compile_prepared_initialization(pf,typeof(f.init),typeof(f.shared))(f.init,f.shared,RK.kernel_prepared_handles(pf))
    ReactiveKernelsNUTSExamples._seed_nuts_children!(f); f
end
for (md,seed) in ((2,20260829),(4,20260829),(4,20260830))
    fr = frame(pf, Float64, md)
    C = ReactiveKernelsNUTSExamples.compile_nuts_native(pf, Fix.nuts_state, Fix.refresh_momentum!!, Fix.nuts!!, fr)
    r = mkrec(seed)
    try
        C.root!(fr, C.scratch, r)
        println("=== md=$md seed=$seed : $(length(r.log)) draws, reached_depth=$(fr.diag.reached_depth) n_steps=$(fr.diag.n_steps) ===")
        for (i,e) in enumerate(r.log); println("  [$i] $(e[1]) = $(e[2])"); end
    catch err
        println("=== md=$md seed=$seed : ERROR $(err) ==="); break
    end
end
println("DRAWLOG2_DONE")
