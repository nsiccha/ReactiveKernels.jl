using ReactiveKernels, LinearAlgebra, Random
const RK = ReactiveKernels
include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","examples","nuts_runtime.jl"))
module Fix; include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","benchmark","nuts_kernel_authoring_fixture.jl")); end

# Recorder (defined in Random so the native domain gate admits it)
Random.eval(quote mutable struct RecRNG <: AbstractRNG; inner::Xoshiro; log::Vector{Any}; end end)
const RecRNG = Random.RecRNG
mkrec(seed) = RecRNG(Random.Xoshiro(seed), Any[])
Random.randn!(r::RecRNG, x::AbstractArray) = (Random.randn!(r.inner, x); push!(r.log,(:m, copy(x))); x)
Base.rand(r::RecRNG, ::Type{Bool}) = (v=Base.rand(r.inner, Bool); push!(r.log,(:d, v)); v)
Random.randexp(r::RecRNG) = (v=Random.randexp(r.inner); push!(r.log,(:e, v)); v)

# ---- THE PRE-GEN BUNDLE CONSTRUCTOR (increment 2) ----
# Pull Xoshiro(seed) in the MAXIMAL interleaved structural draw order:
#   momentum randn D-vec ; then for depth d=1..max_depth: 1 rand(Bool) direction, then d randexp.
# dirs[kd] / exps[ke] align with native's interleaved single-Xoshiro consumption (verified below).
function pregen_bundle(seed::Integer, D::Int, max_depth::Int)
    rng = Random.Xoshiro(seed)
    momentum = Random.randn!(rng, zeros(Float64, D))
    dirs = Bool[]; exps = Float64[]
    for d in 1:max_depth
        push!(dirs, rand(rng, Bool))
        for _ in 1:d; push!(exps, Random.randexp(rng)); end
    end
    (; momentum, dirs, exps)
end

# ---- native oracle setup (as verified) ----
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
    f=RK._construct_nuts_frame(pf,vals(pf,T;metric),md;step_f=RK.partial(Fix.leapfrog!;stepsize=T(.1)),stats_f=Fix.nuts_stats!,min_dham=-1000.0)
    RK.compile_prepared_initialization(pf,typeof(f.init),typeof(f.shared))(f.init,f.shared,RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(f); f
end

# ---- VERIFY: bundle prefix == native's recorded interleaved consumption ----
allok = true
for (md,seed) in ((2,20260829),(4,20260829),(4,20260830),(6,12345),(3,999))
    fr = frame(pf, Float64, md); C = RK.compile_nuts_native(pf, Fix.nuts_state, Fix.refresh_momentum!!, Fix.nuts!!, fr)
    r = mkrec(seed); C.root!(fr, C.scratch, r)
    b = pregen_bundle(seed, 2, md)
    # walk native's recorded log, checking each draw against the bundle by type-counter
    kd=0; ke=0; ok=true
    for e in r.log
        if e[1]==:m; ok &= (e[2] == b.momentum)
        elseif e[1]==:d; kd+=1; ok &= (kd<=length(b.dirs) && e[2]==b.dirs[kd])
        elseif e[1]==:e; ke+=1; ok &= (ke<=length(b.exps) && e[2]==b.exps[ke])
        end
    end
    println("md=$md seed=$seed: native drew ($kd dir, $ke exp), bundle($(length(b.dirs)) dir, $(length(b.exps)) exp) -> parity=$ok")
    global allok &= ok
end
println(allok ? "BUNDLE_VERIFIED: bundle reproduces native's exact draw stream" : "BUNDLE_MISMATCH")
