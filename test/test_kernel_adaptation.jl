# Gates for lowering an AUTHORED free stateful @kernel (dual_averaging_state / welford_var) to a runnable
# object — the AUTHORED recurrence, NOT the package @reactive type (HMC acceptance G3/G4).  Construction and
# method execution share the authoritative plan/ownership/bootstrap substrate; the hot method ABI is concrete,
# inferred, and allocation-free.
using Test, LinearAlgebra, InteractiveUtils
using ReactiveKernels
const RK = ReactiveKernels

module _AdaptFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

_sval(pf, ow, sh, nm) = begin
    PL = RK.kernel_prepared_plan(pf); s = RK.kernel_plan_slot(PL, nm)
    role, slot = RK.kernel_plan_field(PL, s.canon)
    RK._canon_slot(role === :owned ? ow : sh, Val(slot))
end
_scur(pf, ow, sh, nm) = begin
    PL = RK.kernel_prepared_plan(pf); s = RK.kernel_plan_slot(PL, nm)
    role, slot = RK.kernel_plan_field(PL, s.canon)
    RK._canon_current(role === :owned ? ow : sh, Val(slot))
end
_roles(pf) = begin
    PL = RK.kernel_prepared_plan(pf)
    Dict(String(s.path[1]) => RK.kernel_plan_field(PL, s.canon)[1] for s in RK.kernel_plan_slots(PL))
end
_cstate(skel, args...; kwargs...) = begin
    k = RK.compile_stateful(skel, args...; kwargs...)
    k(args...; kwargs...)
end


# ---- executable authored G3/G4 recurrence ---------------------------------------------------------------

_affine(old, new, w) = (one(w) - w) * old + w * new

@inline function _da_batch!(s, x, ::Val{N}) where {N}
    i = 0
    @inbounds while i < N
        RK.stateful_call!(s, Val(:fit!), x)
        i += 1
    end
    nothing
end
@inline function _wv_batch!(s, x, ::Val{N}) where {N}
    i = 0
    @inbounds while i < N
        RK.stateful_call!(s, Val(:step!), x)
        i += 1
    end
    nothing
end
function _hot_alloc(batch, s, x, n)
    batch(s, x, n)
    GC.gc()
    @allocated batch(s, x, n)
end

@testset "kernel_adaptation — G3 DA: every authored field matches the independent recurrence per step" begin
    xs = (0.91, 0.31, 0.63, 0.79, 0.68, 0.54)
    for T in (Float32, Float64)
        init = T(0.65)
        s = _cstate(_AdaptFix.dual_averaging_state, init)
        target, reg, kappa, offset = T(0.8), T(0.05), T(0.75), T(10)
        m, H = one(T), zero(T)
        mu = log(T(10)) + log(init)
        log_final = zero(T)
        for x0 in xs
            x = T(x0)
            m += one(T)
            H += (target - x - H) / (m + offset)
            log_current = mu - sqrt(m) / reg * H
            log_final += m^(-kappa) * (log_current - log_final)
            @inferred RK.stateful_call!(s, Val(:fit!), x)
            @test RK.stateful_get(s, Val(:m)) ≈ m
            @test RK.stateful_get(s, Val(:H)) ≈ H
            @test RK.stateful_get(s, Val(:mu)) ≈ mu
            @test RK.stateful_get(s, Val(:log_current)) ≈ log_current
            @test RK.stateful_get(s, Val(:log_final)) ≈ log_final
            @test RK.stateful_get(s, Val(:current)) ≈ exp(log_current)
            @test RK.stateful_get(s, Val(:final)) ≈ exp(log_final)
        end
        @test @inferred(RK.stateful_get(s, Val(:current))) isa T
        @test isconcretetype(only(Base.return_types(RK.stateful_call!, Tuple{typeof(s),Val{:fit!},T})))
        for n in (Val(64), Val(128), Val(256))
            z = _cstate(_AdaptFix.dual_averaging_state, init)
            @test _hot_alloc(_da_batch!, z, T(0.5), n) == 0
        end
    end
end

function _welford_step(n, mean, var, x, dn)
    n2 = n + dn
    w = dn / n2
    nextmean = _affine.(mean, x, w)
    nextvar = _affine.(var, (x .- nextmean) .* (x .- mean), w)
    n2, nextmean, nextvar
end

@testset "kernel_adaptation — G4 Welford: vector/matrix, default/explicit dn, per-column parity" begin
    for T in (Float32, Float64)
        s = _cstate(_AdaptFix.welford_var, zeros(T, 3))
        n = zero(T); mean = zeros(T, 3); var = zeros(T, 3)
        seq = ((T[1,2,3], one(T)), (T[2,0,4], T(2)), (T[3,1,2], one(T)))
        for (x, dn) in seq
            n, mean, var = _welford_step(n, mean, var, x, dn)
            dn === one(T) ? (@inferred RK.stateful_call!(s, Val(:step!), x)) :
                            (@inferred RK.stateful_call!(s, Val(:step!), x; dn = dn))
            @test RK.stateful_get(s, Val(:n)) ≈ n
            @test RK.stateful_get(s, Val(:mean)) ≈ mean
            @test RK.stateful_get(s, Val(:var)) ≈ var
        end

        mat = T[1 3 5; 2 4 6; 4 2 0]
        for dn in (nothing, T(2))
            sm = _cstate(_AdaptFix.welford_var, zeros(T, 3))
            sv = _cstate(_AdaptFix.welford_var, zeros(T, 3))
            if dn === nothing
                @inferred RK.stateful_call!(sm, Val(:step!), mat)
                for col in eachcol(mat); RK.stateful_call!(sv, Val(:step!), collect(col)); end
            else
                @inferred RK.stateful_call!(sm, Val(:step!), mat; dn = dn)
                for col in eachcol(mat); RK.stateful_call!(sv, Val(:step!), collect(col); dn = dn); end
            end
            for fld in (:n, :mean, :var)
                @test RK.stateful_get(sm, Val(fld)) ≈ RK.stateful_get(sv, Val(fld))
            end
            @test isconcretetype(only(Base.return_types(RK.stateful_call!,
                Tuple{typeof(sm),Val{:step!},Matrix{T}})))
        end
        for (x, batch) in ((T[1,2,3], _wv_batch!), (mat, _wv_batch!)), nrep in (Val(64), Val(128))
            z = _cstate(_AdaptFix.welford_var, zeros(T, 3))
            @test _hot_alloc(batch, z, x, nrep) == 0
        end
    end
end

# No adaptation recurrence depends on public helper-declaration syntax.  Welford's smoothing is captured as
# ordinary arithmetic, and every value call in both MethodIR families is a pure primitive registration.
@testset "kernel_adaptation — DA/Welford authoring has no @rk helper/result/arity contract" begin
    src = read(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"), String)
    wstart = findfirst("@kernel welford_var", src)
    dstart = findfirst("@kernel dual_averaging_state", src)
    dsrc = src[first(dstart):(first(wstart) - 1)]
    wsrc = src[first(wstart):end]
    @test !occursin("@rk_", dsrc)
    @test !occursin("@rk_", wsrc)
    @test !occursin("smooth(", wsrc)
    for skel in (_AdaptFix.dual_averaging_state, _AdaptFix.welford_var), ir in RK.method_irs(skel)
        kinds = Symbol[]
        for st in ir.body
            RK._kmir_walk(st) do node
                node isa RK._RegisteredCall && push!(kinds, node.registration.kind)
            end
        end
        @test !(:declared_effect in kinds)
        @test all(k -> k in (:pure_primitive, :primitive), kinds)
    end
    emit_src = read(joinpath(@__DIR__, "..", "src", "kernel_adaptation.jl"), String)
    @test !occursin("Base.promote_op(", emit_src)
    @test !occursin("Core.Compiler.", emit_src)
    @test !occursin("return_type(", emit_src)
end


# ---- adversarial domain, binder, dispatch, and exception-currentness gates --------------------------------

module _AdaptBad
using ReactiveKernels

@kernel required_kw_state(init) = begin
    value = zero(init)
    step!(x; dn) = value += x * dn
end

@kernel default_order_state(init) = begin
    value = zero(init)
    step!(x; a = x, b = a + one(a)) = value += b
end

@kernel bad_length_state(init) = begin
    value = zero(init)
    step!(x) = value += length(x)
end

@kernel unsupported_result_state(init) = begin
    value = zero(init)
    step!(x) = value += x & x
end

plus_alias = +
@kernel rebound_state(init) = begin
    value = zero(init)
    step!(x) = value = plus_alias(value, x)
end
plus_alias = -

@kernel rhs_throw_state(init) = begin
    value = one(init)
    step!(x) = value += sqrt(x)
end

@kernel cross_method_state(init) = begin
    carried = one(init)
    total = zero(init)
    setcarried!(x) = carried += x
    consume!(x) = total += carried + x
end
end

mutable struct _EvilVector <: AbstractVector{Float64}
    data::Vector{Float64}
    reads::Int
end
Base.size(x::_EvilVector) = size(x.data)
Base.getindex(x::_EvilVector, i::Int) = (x.reads += 1; x.data[i])

mutable struct _EvilMatrix <: AbstractMatrix{Float64}
    data::Matrix{Float64}
    reads::Int
end
Base.size(x::_EvilMatrix) = size(x.data)
Base.getindex(x::_EvilMatrix, i::Int, j::Int) = (x.reads += 1; x.data[i,j])

@testset "kernel_adaptation — exact concrete domains and exhaustive output rules reject spoof surfaces" begin
    w = _cstate(_AdaptFix.welford_var, zeros(3))
    ev = _EvilVector([1.0,2,3], 0); em = _EvilMatrix([1.0 2; 3 4; 5 6], 0)
    @test_throws ArgumentError RK.stateful_call!(w, Val(:step!), ev)
    @test_throws ArgumentError RK.stateful_call!(w, Val(:step!), em)
    @test ev.reads == 0 && em.reads == 0

    badlen = _cstate(_AdaptBad.bad_length_state, 0.0)
    @test_throws ArgumentError RK.stateful_call!(badlen, Val(:step!), 1.0) # length(Float64)
    badout = _cstate(_AdaptBad.unsupported_result_state, 0)
    @test_throws ArgumentError RK.stateful_call!(badout, Val(:step!), 1)   # `&` has no output rule
    @test_throws Exception RK.compile_stateful(_AdaptBad.rebound_state, 0.0)
end

@testset "kernel_adaptation — authoritative keyword binder, including exact matrix kwsplat forwarding" begin
    req = _cstate(_AdaptBad.required_kw_state, 0.0)
    @test_throws UndefKeywordError RK.stateful_call!(req, Val(:step!), 2.0)
    RK.stateful_call!(req, Val(:step!), 2.0; dn = 3.0)
    @test RK.stateful_get(req, Val(:value)) == 6.0
    @test_throws ArgumentError RK.stateful_call!(req, Val(:step!), 2.0; bogus = 3.0)

    ord = _cstate(_AdaptBad.default_order_state, 0.0)
    RK.stateful_call!(ord, Val(:step!), 2.0)
    @test RK.stateful_get(ord, Val(:value)) == 3.0
    RK.stateful_call!(ord, Val(:step!), 2.0; a = 4.0)
    @test RK.stateful_get(ord, Val(:value)) == 8.0
    RK.stateful_call!(ord, Val(:step!), 2.0; a = "unused", b = 5.0)
    @test RK.stateful_get(ord, Val(:value)) == 13.0

    w = _cstate(_AdaptFix.welford_var, zeros(3))
    @test_throws ArgumentError RK.stateful_call!(w, Val(:step!), [1.0,2,3]; bogus = 1.0)
    @test_throws ArgumentError RK.stateful_call!(w, Val(:step!), [1.0 2; 2 3; 3 4]; bogus = 1.0)
    @test_throws ArgumentError RK.stateful_call!(w, Val(:step!), [1.0,2,3]; dn = "bad")
    @test_throws ArgumentError RK.stateful_call!(w, Val(:step!), [1.0 2; 2 3; 3 4]; dn = "bad")
    RK.stateful_call!(w, Val(:step!), [1.0 2; 2 3; 3 4]; dn = 2.0)
    @test RK.stateful_get(w, Val(:n)) == 4.0

    ir = first(x for x in RK.method_irs(_AdaptFix.welford_var) if x.id.decl == 1)
    dup = RK.MethodIR(ir.id, ir.self, (ir.formals..., ir.formals[end]), ir.body, ir.control,
                      ir.effects, ir.resolution_deps, ir.kind, ir.ok, ir.reason, ir.signature)
    @test_throws Exception RK._sm_validate_formals(dup)
end

struct _Pick{S} end
(::_Pick{S})(owned, shared, handles, args, kw) where {S} = S

function _pickset(types)
    arms = Tuple(RK._sm_arm(T, (), (), Tuple{}, _Pick{tag}()) for (T, tag) in types)
    RK._SMSet{:pick,typeof(arms)}(arms)
end

@testset "kernel_adaptation — unique-minimal overload selection is order-independent; ambiguity rejects" begin
    a = _pickset(((Number, :number), (Real, :real)))
    b = _pickset(((Real, :real), (Number, :number)))
    @test RK._sm_dispatch(a, nothing, nothing, nothing, 1.0, NamedTuple()) === :real
    @test RK._sm_dispatch(b, nothing, nothing, nothing, 1.0, NamedTuple()) === :real
    amb = _pickset(((Union{Int,Float64}, :left), (Union{Float64,String}, :right)))
    @test_throws ArgumentError RK._sm_dispatch(amb, nothing, nothing, nothing, 1.0, NamedTuple())
    dup = _pickset(((Real, :one), (Real, :two)))
    @test_throws ArgumentError RK._sm_dispatch(dup, nothing, nothing, nothing, 1.0, NamedTuple())
end

@testset "kernel_adaptation — global method-write union and throw-prefix currentness" begin
    kc = RK.compile_stateful(_AdaptBad.cross_method_state, 1.0)
    c = kc(1.0)
    RK.stateful_call!(c, Val(:setcarried!), 2.0)
    RK.stateful_call!(c, Val(:consume!), 1.0)
    @test RK.stateful_get(c, Val(:carried)) == 3.0
    @test RK.stateful_get(c, Val(:total)) == 4.0
    pfc = getfield(kc, :prepared)
    cp = RK.kernel_plan_slot(RK.kernel_prepared_plan(pfc), :carried)
    role, slot = RK.kernel_plan_field(RK.kernel_prepared_plan(pfc), cp.canon)
    @test role === :owned
    RK._canon_kill!(getfield(c, :owned), Val(slot))
    @test_throws ErrorException RK.stateful_call!(c, Val(:consume!), 1.0) # never reruns carried=one(init)

    kw = RK.compile_stateful(_AdaptFix.welford_var, zeros(3))
    w = kw(zeros(3)); pfw = getfield(kw, :prepared)
    @test_throws DimensionMismatch RK.stateful_call!(w, Val(:step!), [1.0,2.0])
    @test _scur(pfw, getfield(w,:owned), getfield(w,:shared), :n)
    @test !_scur(pfw, getfield(w,:owned), getfield(w,:shared), :var) # killed before materialize!, never blessed
    @test _scur(pfw, getfield(w,:owned), getfield(w,:shared), :mean)

    kr = RK.compile_stateful(_AdaptBad.rhs_throw_state, 1.0)
    r = kr(1.0); pfr = getfield(kr, :prepared)
    @test_throws DomainError RK.stateful_call!(r, Val(:step!), -1.0)
    @test !_scur(pfr, getfield(r,:owned), getfield(r,:shared), :value)

    kd = RK.compile_stateful(_AdaptFix.dual_averaging_state, 0.65)
    d = kd(0.65); pfd = getfield(kd, :prepared)
    ms = RK.kernel_plan_slot(pfd.plan, :m); _, mslot = RK.kernel_plan_field(pfd.plan, ms.canon)
    RK._canon_set!(getfield(d,:owned), Val(mslot), -3.0); RK._canon_bless!(getfield(d,:owned), Val(mslot))
    @test_throws DomainError RK.stateful_call!(d, Val(:fit!), 0.5)
    @test _scur(pfd, getfield(d,:owned), getfield(d,:shared), :m)
    @test _scur(pfd, getfield(d,:owned), getfield(d,:shared), :H)
    @test !_scur(pfd, getfield(d,:owned), getfield(d,:shared), :log_current)
    @test !_scur(pfd, getfield(d,:owned), getfield(d,:shared), :current)
    @test _scur(pfd, getfield(d,:owned), getfield(d,:shared), :log_final) # ensure threw before target kill

    # A forged MethodIR write to a genuinely shared plan slot must be rejected by the emitter itself; it may
    # not infer ownership from the method-local write set.
    baseir = only(RK.method_irs(_AdaptFix.dual_averaging_state))
    forged = RK._PlaceWrite(RK._SelfField((:mu,)), :self, (:mu,), nothing, RK._Lit(1.0), false)
    badir = RK.MethodIR(baseir.id, baseir.self, baseir.formals, (forged,), baseir.control,
        baseir.effects, baseir.resolution_deps, baseir.kind, baseir.ok, baseir.reason, baseir.signature)
    mu = RK.kernel_plan_slot(pfd.plan, :mu).canon
    @test_throws RK._LLowerReject RK.compile_stateful_method(pfd, typeof(getfield(d,:owned)),
        typeof(getfield(d,:shared)), badir, Set((mu,)))
end

function _hot_type_clean(T, seen = Set{Any}())
    T in seen && return true
    push!(seen, T)
    T === Any && return false
    T === Function && return false
    T <: Core.Box && return false
    T <: AbstractDict && return false
    isconcretetype(T) || return false
    T isa DataType || return true
    all(ft -> _hot_type_clean(ft, seen), fieldtypes(T))
end


function _llvm_text(f, tt)
    io = IOBuffer()
    InteractiveUtils.code_llvm(io, f, tt; raw = true, dump_module = false, optimize = true)
    String(take!(io))
end

@testset "kernel_adaptation — typed hot batches contain no generic apply or boxing" begin
    kd = RK.compile_stateful(_AdaptFix.dual_averaging_state, 0.65)
    sd = kd(0.65)
    kw = RK.compile_stateful(_AdaptFix.welford_var, zeros(3))
    sw = kw(zeros(3))
    for (f, tt) in ((_da_batch!, Tuple{typeof(sd),Float64,Val{64}}),
                    (_wv_batch!, Tuple{typeof(sw),Vector{Float64},Val{64}}),
                    (_wv_batch!, Tuple{typeof(sw),Matrix{Float64},Val{64}}))
        llvm = _llvm_text(f, tt)
        @test !occursin("apply_generic", llvm)
        @test !occursin("jl_box", llvm)
    end
end

@testset "kernel_adaptation — compiled hot state is concrete and retains no Dict/Any/Function/Core.Box" begin
    for (k, args) in ((RK.compile_stateful(_AdaptFix.dual_averaging_state, 0.65), (0.65,)),
                      (RK.compile_stateful(_AdaptFix.welford_var, zeros(3)), (zeros(3),)))
        s = k(args...)
        @test _hot_type_clean(typeof(s))
        @test typeof(s) === typeof(k(args...))
    end
end

@testset "kernel_adaptation — authoritative ownership (not a local-seed setdiff)" begin
    # DA: mutated ∪ derived-from-mutated owned; the constant `mu` (from init only) + params stay shared.
    rDA = _roles(RK._prepare_stateful(_AdaptFix.dual_averaging_state))
    @test all(rDA[n] === :owned for n in ("m", "H", "log_current", "log_final", "current", "final"))
    @test all(rDA[n] === :shared for n in ("init", "mu", "target", "regularization_scale", "relaxation_exponent", "offset"))
    # welford: n/mean/var owned (written by step!, incl. the matrix overload's __self__ sibling write which a
    # local seed would miss); template shared. The authoritative closure resolves interprocedural sibling writes.
    rWV = _roles(RK._prepare_stateful(_AdaptFix.welford_var))
    @test rWV["n"] === :owned && rWV["mean"] === :owned && rWV["var"] === :owned && rWV["template"] === :shared
end

@testset "kernel_adaptation — dual_averaging_state: single-pass construct matches the authored initializers" begin
    da = _AdaptFix.dual_averaging_state
    pf = RK._prepare_stateful(da)
    for (T, tgt) in ((Float64, 0.8), (Float32, 0.8f0))
        ow, sh = RK._construct_stateful(da, pf, T(tgt))
        mu_ref = log(T(10)) + log(T(tgt))
        @test _sval(pf, ow, sh, :m) === one(T)                      # m = one(init), typed
        @test _sval(pf, ow, sh, :H) === zero(T)                     # H = zero(init)
        @test _sval(pf, ow, sh, :mu) ≈ mu_ref                       # mu = log(10) + log(init)
        @test _sval(pf, ow, sh, :log_current) ≈ mu_ref              # log_current = mu - sqrt(m)/reg*H  (H=0)
        @test _sval(pf, ow, sh, :log_final) === zero(T)
        @test _sval(pf, ow, sh, :current) ≈ exp(mu_ref)             # current = exp(log_current)  (exp admitted)
        @test _sval(pf, ow, sh, :final) === one(T)                  # final = exp(log_final) = exp(0) = 1
        @test typeof(_sval(pf, ow, sh, :current)) === T             # concrete init-derived type
        @test all(_scur(pf, ow, sh, n) for n in (:m, :H, :mu, :log_current, :log_final, :current, :final))  # all current
    end
    # keyword override flows through the authoritative binder; unknown / extra reject (never silently ignored)
    ow2, sh2 = RK._construct_stateful(da, pf, 0.8; target = 0.9, offset = 20.0)
    @test _sval(pf, ow2, sh2, :target) == 0.9 && _sval(pf, ow2, sh2, :offset) == 20.0
    @test_throws ArgumentError RK._construct_stateful(da, pf, 0.8; bogus = 1)   # unknown keyword
    @test_throws MethodError RK._construct_stateful(da, pf, 0.8, 0.9)           # extra positional
    # @inferred + repeated same-signature CONCRETE-TYPE IDENTITY (deterministic layout, no boxing)
    @inferred RK._construct_stateful(da, pf, 0.8)
    a = RK._construct_stateful(da, pf, 0.8); b = RK._construct_stateful(da, pf, 0.8)
    @test typeof(a) === typeof(b) && isconcretetype(typeof(a[1])) && isconcretetype(typeof(a[2]))
end

@testset "kernel_adaptation — welford_var: single-pass construct + EXECUTE-ONCE (buffer-identity witness)" begin
    wv = _AdaptFix.welford_var
    pf = RK._prepare_stateful(wv)
    for T in (Float64, Float32)
        ow, sh = RK._construct_stateful(wv, pf, T[0, 0, 0])
        @test _sval(pf, ow, sh, :n) === zero(T)
        @test _sval(pf, ow, sh, :mean) == T[0, 0, 0] && _sval(pf, ow, sh, :var) == T[0, 0, 0]
        @test eltype(_sval(pf, ow, sh, :mean)) === T
        @test _sval(pf, ow, sh, :mean) !== _sval(pf, ow, sh, :var)             # distinct fresh buffers
        @test all(_scur(pf, ow, sh, n) for n in (:n, :mean, :var, :template))
    end
    @inferred RK._construct_stateful(wv, pf, [0.0, 0.0, 0.0])
    a = RK._construct_stateful(wv, pf, [0.0, 0.0, 0.0]); b = RK._construct_stateful(wv, pf, [0.0, 0.0, 0.0])
    @test typeof(a) === typeof(b)
    @test _sval(pf, a[1], a[2], :mean) !== _sval(pf, b[1], b[2], :mean)        # per-instance isolation
    # EXECUTE-ONCE proof: the construct path is the ACCEPTED cold bootstrap (whose per-recipe execute-once is
    # counter-proven NUTS-side by cg.n==1) — NO second initializer executor. Witness it directly: a produced
    # field buffer is the SINGLE bootstrap's output BY IDENTITY (a second executor would replace it with a
    # different buffer). A per-recipe counter on these PURE initializers is precluded by the (correct) domain
    # gate (custom/impure scalar types are rejected), so identity-retention is the equivalent execute-once proof.
    plan = RK.kernel_prepared_plan(pf); H = RK.kernel_prepared_handles(pf)
    src = RK._stateful_sources(wv, pf, ([1.0, 2.0, 3.0],), NamedTuple())
    cvals = RK._bootstrap_canon_values(plan, H, src)                            # ONE bootstrap execution
    ow, sh = RK._construct_endpoint_from_values(plan, H, cvals)
    canons, _ = RK._plan_superset_from_key(RK.kernel_plan_key(plan))
    mslot = RK.kernel_plan_field(plan, RK.kernel_plan_slot(plan, :mean).canon)[2]
    vslot = RK.kernel_plan_field(plan, RK.kernel_plan_slot(plan, :var).canon)[2]
    mi = findfirst(==(RK.kernel_plan_slot(plan, :mean).canon), canons)
    vi = findfirst(==(RK.kernel_plan_slot(plan, :var).canon), canons)
    @test RK._canon_slot(ow, Val(mslot)) === cvals[mi]                          # produced buffer IS the bootstrap output
    @test RK._canon_slot(ow, Val(vslot)) === cvals[vi]
end
