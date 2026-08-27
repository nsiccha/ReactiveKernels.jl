# Executable prepared-endpoint phasepoint receipt (RK 07:43 joint milestone). The interim compile_leaf /
# effect-root / copy / transition scaffold tests (which consumed fake `_OwnerState`/`_Synth` storage) were
# retired with that scaffold on the 4c6ed92 seam; the generic schedule-MODEL tests live in
# test_kernel_lowering.jl (no fake storage). This suite executes the REAL captured six-handle phasepoint.
using ReactiveKernels, Test, LinearAlgebra, Random
const RK = ReactiveKernels

# ============================================================================================
# EXECUTABLE PREPARED-ENDPOINT phasepoint — the RK 07:43 joint receipt (six real captured handles).
# Isolated module so the endpoint @kernel globals never collide with other suites' fixtures. Canonical
# ids are resolved by FIELD NAME through the plan (the global Value-id counter is not 1..N in-suite).
# ============================================================================================
module _PPFix
    using ReactiveKernels, LinearAlgebra
    @kernel euclidean_ep(grad_f, metric, pos, mom) = begin
        pot, dpot_dpos = grad_f(pos)
        chol_metric = cholesky(metric)
        dkin_dmom = chol_metric \ mom
        kin = oftype(pot, 0.5) * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
        ham = pot + kin
        dham_dpos = dpot_dpos
        dham_dmom = dkin_dmom
    end
    # the FINAL ccb35d3 leapfrog!: typed half-kick / drift / half-kick (RK 08:49 acceptance discriminator)
    @kernel leapfrog!(phasepoint; stepsize) = begin
        @. phasepoint.mom -= oftype(stepsize, 0.5) * stepsize * phasepoint.dham_dpos
        @. phasepoint.pos +=                       stepsize * phasepoint.dham_dmom
        @. phasepoint.mom -= oftype(stepsize, 0.5) * stepsize * phasepoint.dham_dpos
    end
    mutable struct CountPgrad; n::Int; end
    (c::CountPgrad)(dest, pos) = (c.n += 1; dest .= 2 .* pos; sum(abs2, pos))
    # a grad that throws on its Nth call — for executed-prefix exception soundness
    mutable struct ThrowOnGrad; n::Int; throw_at::Int; end
    (g::ThrowOnGrad)(dest, pos) = (g.n += 1; g.n >= g.throw_at && error("grad boom"); dest .= 2 .* pos; sum(abs2, pos))
end

using LinearAlgebra: cholesky, ldiv!, dot, logdet, Cholesky, I, PosDefException

_pp_pf() = RK._prepare_factory(_PPFix.euclidean_ep, RK.kernel_registration(_PPFix.leapfrog!))
# canonical id of a plan field by NAME; @node resolved by its gensym prefix.
_pp_c(plan, name::Symbol) = only(s.canon for s in RK.kernel_plan_slots(plan) if s.path[end] === name)
_pp_node(plan) = only(unique(s.canon for s in RK.kernel_plan_slots(plan) if startswith(String(s.path[end]), "##node")))
# UNCOMPUTED typed placeholders keyed by REAL canons (RK 08:03/08:15): allocated factors Matrix + the
# Cholesky wrapper constructor (Char 'U') WITHOUT calling cholesky; zeros elsewhere. NO recipe is executed.
function _pp_values(plan, ::Type{T}, d, pgrad) where {T}
    v = Dict{Int,Any}()
    for s in RK.kernel_plan_slots(plan)
        haskey(v, s.canon) && continue
        nm = s.path[end]; ns = String(nm)
        v[s.canon] =
            nm === :grad_f      ? pgrad :
            nm === :metric      ? (Matrix{T}(I(d)) .+ T(0.5)) :
            nm === :pos         ? T[1,2,3][1:d] :
            nm === :mom         ? T[0.5,-1,2][1:d] :
            nm === :chol_metric ? Cholesky(Matrix{T}(undef, d, d), 'U', 0) :
            startswith(ns, "##node") ? zero(T) :
            (nm in (:pot, :kin, :ham)) ? zero(T) :
            zeros(T, d)
    end
    v
end
_pp_construct(pf, plan, values) = RK._kernel_construct_endpoint(
    Val(RK.kernel_prepared_token(pf)), plan, values, RK.kernel_prepared_external(pf))
_pp_rd(plan, ow, sh, c) = (fs = RK.kernel_plan_field(plan, c);
    fs[1] === :owned ? RK._canon_slot(ow, Val(fs[2])) : RK._canon_slot(sh, Val(fs[2])))
_pp_cur(plan, ow, sh, c) = (fs = RK.kernel_plan_field(plan, c);
    RK._canon_current(fs[1] === :owned ? ow : sh, Val(fs[2])))
function _pp_ref(::Type{T}, d) where {T}
    metric = Matrix{T}(I(d)) .+ T(0.5); pos = T[1,2,3][1:d]; mom = T[0.5,-1,2][1:d]
    ch = cholesky(metric); dk = ch \ mom
    (pot = sum(abs2, pos), dpot = 2 .* pos, ld = logdet(ch), dkin = dk,
     kin = T(0.5)*(logdet(ch)+dot(mom,dk)), ham = sum(abs2,pos)+T(0.5)*(logdet(ch)+dot(mom,dk)))
end

# A Diagonal-typed construction with arbitrary dimension/data. The Cholesky value is a typed, uncomputed
# placeholder; the executable prepared initialization must replace it before the emitted ldiv handle runs.
function _pp_diag_values(plan, ::Type{T}, diag, pos, mom, pgrad) where {T}
    v = Dict{Int,Any}(); d = length(diag)
    for s in RK.kernel_plan_slots(plan)
        haskey(v, s.canon) && continue
        nm = s.path[end]; ns = String(nm)
        v[s.canon] =
            nm === :grad_f      ? pgrad :
            nm === :metric      ? Diagonal(copy(diag)) :
            nm === :pos         ? copy(pos) :
            nm === :mom         ? copy(mom) :
            nm === :chol_metric ? Cholesky(Diagonal(Vector{T}(undef, d)), 'U', 0) :
            startswith(ns, "##node") ? zero(T) :
            (nm in (:pot, :kin, :ham)) ? zero(T) :
            zeros(T, size(nm in (:dkin_dmom, :dham_dmom) ? mom : pos))
    end
    v
end

_pp_bytes_equal(a::Vector{T}, b::Vector{T}) where {T<:Union{Float32,Float64}} =
    reinterpret(UInt8, a) == reinterpret(UInt8, b)

@testset "prepared-endpoint — FULL six-handle cold init: math + exact masks + counts, F32 & F64 (RK 08:04b)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf)
    @test length(hs) == 6
    @test RK.kernel_plan_recipes(plan) == (2, 3, 1, 4, 5, 6)
    @test [RK.recipe_handle_mode(h) for h in hs] == [:destination, :assign, :assign, :ldiv, :assign, :assign]
    for T in (Float64, Float32)
        d = 3; cp = _PPFix.CountPgrad(0)
        ow, sh = _pp_construct(pf, plan, _pp_values(plan, T, d, cp))
        @test cp.n == 0                                        # construction computes NOTHING (RK 08:00 gate 1)
        @test RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs) === ow  # returns owned
        r = _pp_ref(T, d)
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :pot))       ≈ r.pot
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :dpot_dpos)) ≈ r.dpot
        @test _pp_rd(plan, ow, sh, _pp_node(plan))          ≈ r.ld
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :dkin_dmom)) ≈ r.dkin
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :kin))       ≈ r.kin
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :ham))       ≈ r.ham
        @test cp.n == 1                                        # EXACTLY one pgrad (destination handle)
        cur = Set{Int}(s.canon for s in RK.kernel_plan_slots(plan) if _pp_cur(plan, ow, sh, s.canon))
        @test cur == Set(RK.kernel_plan_entry_current(plan))   # final currentness == entry_current EXACTLY
    end
end

@testset "prepared-endpoint — Diagonal Cholesky ldiv is two-factor exact; dense lowering unchanged" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf)
    li = only(i for i in eachindex(hs) if RK.recipe_handle_mode(hs[i]) === :ldiv)
    lh = hs[li]; ins = collect(lh.inputs); outs = collect(lh.outputs)

    # The discriminator is solely the concrete prepared factor-slot TYPE. Float16, dense, and a Diagonal over
    # non-builtin backing storage remain on the original generic lowering.
    @test RK._pp_diag_cholesky_ldiv_type(Cholesky{Float64,Diagonal{Float64,Vector{Float64}}},
                                         Vector{Float64},Vector{Float64})
    @test RK._pp_diag_cholesky_ldiv_type(Cholesky{Float32,Diagonal{Float32,Vector{Float32}}},
                                         Vector{Float32},Vector{Float32})
    @test !RK._pp_diag_cholesky_ldiv_type(Cholesky{Float16,Diagonal{Float16,Vector{Float16}}},
                                          Vector{Float16},Vector{Float16})
    @test !RK._pp_diag_cholesky_ldiv_type(Cholesky{Float64,Matrix{Float64}},
                                          Vector{Float64},Vector{Float64})
    @test !RK._pp_diag_cholesky_ldiv_type(
        Cholesky{Float64,Diagonal{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}},
        Vector{Float64},Vector{Float64})
    @test !RK._pp_diag_cholesky_ldiv_type(Cholesky{Float64,Diagonal{Float64,Vector{Float64}}},
                                          Array{Float64,3},Array{Float64,3})
    @test !RK._pp_diag_cholesky_ldiv_type(Cholesky{Float64,Diagonal{Float64,Vector{Float64}}},
                                          Vector{Float64},Matrix{Float64})

    # Source/lowering pin: dense remains byte-structurally the old ONE `ldiv!(dest,factor,rhs)` statement.
    od, sd = _pp_construct(pf, plan, _pp_values(plan, Float64, 3, _PPFix.CountPgrad(0)))
    dense = Any[]; RK._pp_emit_handle!(dense, plan, lh, li, typeof(od), typeof(sd))
    expected_dense = :(ldiv!($(RK._pp_read(plan, outs[1])), $(RK._pp_read(plan, ins[1])), $(RK._pp_read(plan, ins[2]))))
    @test dense[1] == expected_dense
    @test length(dense[1].args) == 4                         # callee + dest + factor + rhs
    @test !any(x -> x isa Expr && x.head === :call && x.args[1] === :copyto!, dense)

    # The admitted Diagonal type emits the guarded two-factor helper, then the existing bless. The helper catches
    # the tempting but mathematically wrong one-factor lowering and owns every IEEE/alias generic fallback.
    dg = ones(Float64, 3); pos = ones(Float64, 3); mom = ones(Float64, 3)
    og, sg = _pp_construct(pf, plan, _pp_diag_values(plan, Float64, dg, pos, mom, _PPFix.CountPgrad(0)))
    diagonal = Any[]; RK._pp_emit_handle!(diagonal, plan, lh, li, typeof(og), typeof(sg))
    @test diagonal[1] == :(_pp_diag_cholesky_ldiv!($(RK._pp_read(plan, outs[1])),
                                                    $(RK._pp_read(plan, ins[1])),
                                                    $(RK._pp_read(plan, ins[2]))))
    @test count(x -> x isa Expr && x.head === :call && x.args[1] === :_pp_diag_cholesky_ldiv!,
                diagonal) == 1

    # Executable exactness against the generic `F \\ rhs`: F32/F64, dimensions 1/5/17, unit + scaled, 20 seeds.
    for T in (Float64, Float32), d in (1, 5, 17), scaled in (false, true), seed in 1:20
        rng = Random.Xoshiro(seed + 100d + (scaled ? 1000 : 0))
        diag = scaled ? T.(0.25 .+ 4 .* rand(rng, T, d)) : ones(T, d)
        pos = randn(rng, T, d); mom = randn(rng, T, d); cp = _PPFix.CountPgrad(0)
        ow, sh = _pp_construct(pf, plan, _pp_diag_values(plan, T, diag, pos, mom, cp))
        init = RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))
        @test init(ow, sh, hs) === ow
        F = cholesky(Diagonal(diag)); reference = F \ mom
        got = _pp_rd(plan, ow, sh, _pp_c(plan, :dkin_dmom))
        @test _pp_bytes_equal(got, reference)
        @test cp.n == 1
    end

    # Failure prefix: the rhs is copied, the first factor solve rejects the dimension, and the ldiv output is
    # never blessed. This is the same executed-prefix/currentness contract as generic Factorization ldiv!.
    T = Float64; diag = T[1, 4, 9]; pos = T[1, 2, 3]; mom = T[4, 5, 6, 7]
    ow, sh = _pp_construct(pf, plan, _pp_diag_values(plan, T, diag, pos, mom, _PPFix.CountPgrad(0)))
    candidate_error = try
        RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs); nothing
    catch e
        e
    end
    generic_dest = zeros(T, length(mom)); generic_error = try
        ldiv!(generic_dest, cholesky(Diagonal(diag)), mom); nothing
    catch e
        e
    end
    @test candidate_error isa DimensionMismatch
    @test typeof(candidate_error) === typeof(generic_error)
    @test _pp_rd(plan, ow, sh, _pp_c(plan, :dkin_dmom)) == generic_dest == mom
    @test !_pp_cur(plan, ow, sh, _pp_c(plan, :dkin_dmom))
end

function _pp_ldiv_outcome(make, solve)
    dest, factor, rhs = make(); err = try
        solve(dest, factor, rhs); nothing
    catch e
        e
    end
    bytes(x) = copy(reinterpret(UInt8, vec(x)))
    (error_type=err === nothing ? nothing : typeof(err),
     error_text=err === nothing ? nothing : sprint(showerror,err),
     dest=bytes(dest),factor=bytes(factor.factors.diag),rhs=bytes(rhs))
end

@testset "prepared-endpoint — Diagonal fast helper IEEE/alias adversaries match generic value + throw prefix" begin
    for T in (Float64,Float32)
        tiny=nextfloat(zero(T)); big=floatmax(T)
        scenarios = (
            # finite/nonzero inputs whose first division overflows; post-check must replay generic from intact rhs
            () -> (zeros(T,1),Cholesky(Diagonal(T[tiny]),'U',0),T[one(T)]),
            () -> (zeros(T,1),Cholesky(Diagonal(T[big]),'U',0),T[tiny]), # underflow/zero result
            () -> (zeros(T,2),Cholesky(Diagonal(T[1,2]),'U',0),T[-zero(T),one(T)]),
            () -> (zeros(T,2),Cholesky(Diagonal(T[1,2]),'U',0),T[Inf,one(T)]),
            () -> (zeros(T,2),Cholesky(Diagonal(T[1,2]),'U',0),T[NaN,one(T)]),
            () -> (zeros(T,2),Cholesky(Diagonal(T[0,2]),'U',0),T[one(T),one(T)]),
            () -> (zeros(T,2),Cholesky(Diagonal(T[Inf,2]),'U',0),T[one(T),one(T)]),
            () -> (zeros(T,2),Cholesky(Diagonal(T[NaN,2]),'U',0),T[one(T),one(T)]),
            () -> (zeros(T,3),Cholesky(Diagonal(T[1,2]),'U',0),T[1,2,3]), # length mismatch
            () -> begin x=T[3,4]; (x,Cholesky(Diagonal(T[1,2]),'U',0),x) end, # dest === rhs
            () -> begin d=T[1,2]; (d,Cholesky(Diagonal(d),'U',0),T[3,4]) end, # dest === factor backing
            () -> begin r=T[1,2]; (zeros(T,2),Cholesky(Diagonal(r),'U',0),r) end, # rhs === factor backing
            # Object identity is insufficient: distinct builtin Vectors can share the same storage.
            () -> begin
                r=T[3,4]; d=unsafe_wrap(Vector{T},pointer(r),length(r);own=false)
                @assert d !== r && Base.mightalias(d,r)
                (d,Cholesky(Diagonal(T[1,2]),'U',0),r)
            end,
            () -> begin
                b=T[1,2]; d=unsafe_wrap(Vector{T},pointer(b),length(b);own=false)
                @assert d !== b && Base.mightalias(d,b)
                (d,Cholesky(Diagonal(b),'U',0),T[3,4])
            end,
            () -> begin
                b=T[1,2]; r=unsafe_wrap(Vector{T},pointer(b),length(b);own=false)
                @assert r !== b && Base.mightalias(r,b)
                (zeros(T,2),Cholesky(Diagonal(b),'U',0),r)
            end,
            # `Base.mightalias` misses partially overlapping unsafe-wrapped builtin Vectors when their start
            # pointers differ.  Exercise each operand pair so the helper's raw contiguous-range guard is pinned.
            () -> begin
                b=T[1,2,3,4]; d=unsafe_wrap(Vector{T},pointer(b,2),3;own=false)
                f=unsafe_wrap(Vector{T},pointer(b),3;own=false)
                @assert !Base.mightalias(d,f) && RK._pp_vector_overlaps(d,f)
                (d,Cholesky(Diagonal(f),'U',0),T[5,6,7])
            end,
            () -> begin
                b=T[3,4,5,6]; d=unsafe_wrap(Vector{T},pointer(b,2),3;own=false)
                r=unsafe_wrap(Vector{T},pointer(b),3;own=false)
                @assert !Base.mightalias(d,r) && RK._pp_vector_overlaps(d,r)
                (d,Cholesky(Diagonal(T[1,2,3]),'U',0),r)
            end,
            () -> begin
                b=T[1,2,3,4]; r=unsafe_wrap(Vector{T},pointer(b,2),3;own=false)
                f=unsafe_wrap(Vector{T},pointer(b),3;own=false)
                @assert !Base.mightalias(r,f) && RK._pp_vector_overlaps(r,f)
                (zeros(T,3),Cholesky(Diagonal(f),'U',0),r)
            end
        )
        for make in scenarios
            fast=_pp_ldiv_outcome(make,RK._pp_diag_cholesky_ldiv!)
            generic=_pp_ldiv_outcome(make,(d,f,r)->ldiv!(d,f,r))
            @test fast == generic
        end

        # Array{T,3} is outside the fast helper ABI, and the emitter stays byte-structurally on generic ldiv!.
        pf=_pp_pf(); plan=RK.kernel_prepared_plan(pf); hs=RK.kernel_prepared_handles(pf)
        li=only(i for i in eachindex(hs) if RK.recipe_handle_mode(hs[i]) === :ldiv); lh=hs[li]
        diag=T[1,2]; pos=T[1,2]; mom=reshape(T[3,4],2,1,1)
        ow,sh=_pp_construct(pf,plan,_pp_diag_values(plan,T,diag,pos,mom,_PPFix.CountPgrad(0)))
        emitted=Any[];RK._pp_emit_handle!(emitted,plan,lh,li,typeof(ow),typeof(sh))
        @test emitted[1].args[1] === :ldiv! && length(emitted[1].args) == 4
        candidate_error=try
            RK.compile_prepared_initialization(pf,typeof(ow),typeof(sh))(ow,sh,hs);nothing
        catch e
            e
        end
        dest=zeros(T,size(mom));generic_error=try
            ldiv!(dest,cholesky(Diagonal(diag)),mom);nothing
        catch e
            e
        end
        @test typeof(candidate_error) === typeof(generic_error) === MethodError
        @test vec(_pp_rd(plan,ow,sh,_pp_c(plan,:dkin_dmom))) == vec(dest)
        @test !_pp_cur(plan,ow,sh,_pp_c(plan,:dkin_dmom))
    end
end

@testset "prepared-endpoint — transition trace from MethodIR writes: grad/velocity/kin/ham, chol+@node EXCLUDED (RK 08:06/08:10/08:13)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf)
    tr = RK.prepared_transition_trace(plan, RK.method_irs(_PPFix.leapfrog!)[1])   # PRODUCTION from leaf writes
    Rids = RK.selected_trace_recipes(tr)                       # read-only accessor (RK 08:13; no type introspection)
    @test RK.selected_trace_key(tr) === RK.kernel_plan_key(plan)
    prod = Dict(c => r for (c, r) in RK.kernel_plan_producer(plan))
    grad_r, vel_r = prod[_pp_c(plan, :pot)], prod[_pp_c(plan, :dkin_dmom)]
    kin_r, ham_r  = prod[_pp_c(plan, :kin)], prod[_pp_c(plan, :ham)]
    chol_r, node_r = prod[_pp_c(plan, :chol_metric)], prod[_pp_node(plan)]
    @test Set(Rids) == Set([grad_r, vel_r, kin_r, ham_r])      # grad/velocity/kin/ham by RECIPE IDENTITY
    @test !(chol_r in Rids) && !(node_r in Rids)               # exact chol + @node recipe ids EXCLUDED
    # PROVENANCE (RK 08:37): the PUBLIC production API takes the leaf MethodIR and derives the trace itself,
    # so a caller CANNOT inject a recipe list — a _SelectedTrace is not an accepted public argument.
    @test_throws MethodError RK.compile_prepared_schedule(pf, RK._CanonOwned1, RK._CanonShared1,
        RK._SelectedTrace{RK.kernel_plan_key(plan), (grad_r, vel_r)}())
    # A same-Key / ARBITRARY-Rids trace IS internally constructible (compiler-internal provenance, NOT a
    # security boundary) — but reachable only through the PRIVATE overload, which is test-only. It compiles
    # whatever Rids it is handed; the public path above never exposes that lever.
    forged = RK._SelectedTrace{RK.kernel_plan_key(plan), (chol_r,)}()   # wrong recipe, right Key
    @test RK.selected_trace_recipes(forged) == (chol_r,)               # constructible; no boundary
    # the private overload still rejects a WRONG-plan Key (the one real provenance check it enforces)
    @test_throws RK._LLowerReject RK._compile_prepared_schedule(pf, RK._CanonOwned1,
        RK._CanonShared1, RK._SelectedTrace{:forged_key, (2, 4)}())
end

@testset "prepared-endpoint — POST-WRITE RECOMPUTE 0-B/@inferred, chol+@node Δ0, pgrad Δ1, F32 & F64 (RK 08:04c/08:15)" begin
    for T in (Float64, Float32)
        pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf)
        d = 3; cp = _PPFix.CountPgrad(0)
        ow, sh = _pp_construct(pf, plan, _pp_values(plan, T, d, cp))
        RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs)
        @test cp.n == 1
        warm = RK.compile_prepared_schedule(pf, typeof(ow), typeof(sh), RK.method_irs(_PPFix.leapfrog!)[1])
        @test warm(ow, sh, hs) === ow                          # returns the owned object (no tuple alloc)
        chol0 = _pp_rd(plan, ow, sh, _pp_c(plan, :chol_metric)); node0 = _pp_rd(plan, ow, sh, _pp_node(plan))
        n0 = cp.n
        @test (@allocated warm(ow, sh, hs)) == 0               # EXACT 0-B warmed post-write recompute
        @test cp.n == n0 + 1                                   # pgrad Δ one per post-write recompute
        Test.@inferred warm(ow, sh, hs)
        # chol/@node Δ ZERO: same object identity AND value (statically omitted, not guarded)
        warm(ow, sh, hs)
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :chol_metric)) === chol0    # SAME Cholesky object
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :chol_metric)).factors == chol0.factors
        @test _pp_rd(plan, ow, sh, _pp_node(plan)) == node0
    end
end

@testset "prepared-endpoint — JOINT poison gate: live-graph op/producer mutation cannot change captured exec (RK 08:15)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf); d = 3
    cp = _PPFix.CountPgrad(0)
    ow, sh = _pp_construct(pf, plan, _pp_values(plan, Float64, d, cp))
    init = RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))
    leaf = RK.method_irs(_PPFix.leapfrog!)[1]
    warm = RK.compile_prepared_schedule(pf, typeof(ow), typeof(sh), leaf)
    init(ow, sh, hs); ham_clean = _pp_rd(plan, ow, sh, _pp_c(plan, :ham))
    trace_clean = RK.selected_trace_recipes(RK.prepared_transition_trace(plan, leaf))
    g = RK.kernel_graph(_PPFix.euclidean_ep)
    saved_recipes = copy(g.recipes); saved_producers = copy(g.producers)
    try
        empty!(g.recipes); empty!(g.producers)                 # POISON the live graph after preparation
        ow2, sh2 = _pp_construct(pf, plan, _pp_values(plan, Float64, d, _PPFix.CountPgrad(0)))
        init(ow2, sh2, hs)                                     # already-captured executor — no live-graph read
        @test _pp_rd(plan, ow2, sh2, _pp_c(plan, :ham)) ≈ ham_clean            # identical math
        tr2 = RK.prepared_transition_trace(plan, RK.method_irs(_PPFix.leapfrog!)[1])
        @test RK.selected_trace_recipes(tr2) == trace_clean   # identical trace (plan-derived, not graph)
    finally
        append!(g.recipes, saved_recipes); merge!(g.producers, saved_producers)
    end
end

@testset "prepared-endpoint — exception soundness: pgrad throw + genuine later-handle throw (RK 08:00/08:07)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf); d = 3
    cpot, cdpot = _pp_c(plan, :pot), _pp_c(plan, :dpot_dpos); cchol = _pp_c(plan, :chol_metric)
    # (1) pgrad (handle 1) THROWS -> pot+dpot BOTH dirty
    throwgrad(dest, pos) = error("pgrad boom")
    ow, sh = _pp_construct(pf, plan, _pp_values(plan, Float64, d, throwgrad))
    init = RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))
    @test_throws ErrorException init(ow, sh, hs)
    @test !_pp_cur(plan, ow, sh, cpot) && !_pp_cur(plan, ow, sh, cdpot)
    # (2) GENUINE later-handle throw: cholesky (handle 2) on an INDEFINITE metric throws AFTER grad succeeds
    cp = _PPFix.CountPgrad(0); v2 = _pp_values(plan, Float64, d, cp)
    v2[_pp_c(plan, :metric)] = [1.0 2.0 3.0; 2.0 1.0 4.0; 3.0 4.0 1.0]   # symmetric INDEFINITE -> PosDefException
    ow2, sh2 = _pp_construct(pf, plan, v2)
    @test_throws PosDefException init(ow2, sh2, hs)
    @test cp.n == 1                                            # grad ran & completed
    @test _pp_cur(plan, ow2, sh2, cpot) && _pp_cur(plan, ow2, sh2, cdpot)  # grad outputs BLESSED
    @test !_pp_cur(plan, ow2, sh2, cchol)                      # chol output NOT blessed (dirty); obj discardable
end

@testset "prepared-endpoint — child copy: owned-only children, SAME shared authority, ZERO extra pgrad, F32 & F64 (RK 06:53/08:15)" begin
    for T in (Float64, Float32)
        pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf); d = 3
        cham = _pp_c(plan, :ham)
        cp = _PPFix.CountPgrad(0)
        ow, sh = _pp_construct(pf, plan, _pp_values(plan, T, d, cp))
        RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs)
        @test cp.n == 1
        tok = RK.kernel_prepared_token(pf)
        fwd = RK._kernel_construct_owned_child(Val(tok), plan, _pp_values(plan, T, d, cp))
        bwd = RK._kernel_construct_owned_child(Val(tok), plan, _pp_values(plan, T, d, cp))
        @test typeof(fwd) == typeof(ow) && typeof(bwd) == typeof(ow)   # owned layout, NO shared authority built
        RK._seed_children!(ow, fwd, bwd)
        @test cp.n == 1                                        # ZERO additional pgrad across child seeds
        @test RK._canon_slot(fwd, Val(RK.kernel_plan_field(plan, cham)[2])) ≈
              RK._canon_slot(ow,  Val(RK.kernel_plan_field(plan, cham)[2]))   # ham transferred
        # EXECUTE the child against the SAME shared authority `sh` (fixed metric) — proves shared wiring:
        warm = RK.compile_prepared_schedule(pf, typeof(fwd), typeof(sh), RK.method_irs(_PPFix.leapfrog!)[1])
        @test warm(fwd, sh, hs) === fwd                       # child recompute returns the child owned obj
        warm(fwd, sh, hs)
        @test (@allocated warm(fwd, sh, hs)) == 0             # child post-write recompute 0-B over the shared chol
        @test _pp_rd(plan, fwd, sh, cham) ≈ _pp_ref(T, d).ham # child recomputes correct ham via shared chol
    end
end

# ---- consume the REAL ccb35d3 benchmark fixture (RK 08:59: the positive receipt must be non-vacuous) ----
module _FixLF
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end
const _FIXLF = _FixLF.leapfrog!

@testset "executable leapfrog — REAL ccb35d3 fixture leapfrog! composed with prepared recompute (RK 08:42/08:45/08:51/08:59)" begin
    pf = RK._prepare_factory(_PPFix.euclidean_ep, RK.kernel_registration(_FIXLF)); plan = RK.kernel_prepared_plan(pf)
    hs = RK.kernel_prepared_handles(pf)
    leaf = RK.method_irs(_FIXLF)[1]
    @test length(RK._exec_place_writes(leaf)) == 3               # half-kick / drift / half-kick
    # TIE (non-vacuous): the phasepoint-receipt _PPFix.leapfrog! is byte-identical in WRITE STRUCTURE to the
    # real fixture leapfrog! — same targets/order, so the whole receipt is anchored to ccb35d3.
    ppw = RK._exec_place_writes(RK.method_irs(_PPFix.leapfrog!)[1])
    @test length(ppw) == length(RK._exec_place_writes(leaf))
    @test [w.target for w in ppw] == [w.target for w in RK._exec_place_writes(leaf)]
    for T in (Float64, Float32)
        d = 3; cp = _PPFix.CountPgrad(0)
        ow, sh = _pp_construct(pf, plan, _pp_values(plan, T, d, cp))
        RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs)   # cold init (one pgrad)
        @test cp.n == 1
        lf = RK.compile_leapfrog(pf, typeof(ow), typeof(sh), leaf)
        pos0 = copy(_pp_rd(plan, ow, sh, _pp_c(plan, :pos))); mom0 = copy(_pp_rd(plan, ow, sh, _pp_c(plan, :mom)))
        h = T(0.1); n0 = cp.n
        @test lf(ow, sh, hs, (stepsize = h,)) === ow             # !!-style: returns owned
        @test cp.n == n0 + 1                                     # EXACTLY one pgrad per actual leaf
        # ANALYTIC values (Gaussian pot=|pos|^2, grad=2pos, metric M): one leapfrog step in closed form
        M = Matrix{T}(I(d)) .+ T(0.5)
        mom_h = mom0 .- h .* pos0
        pos_a = pos0 .+ h .* (M \ mom_h)
        mom_a = mom_h .- h .* pos_a
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :pos)) ≈ pos_a
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :mom)) ≈ mom_a
        # runtime masks: gradient (mom/pos/dpot/pot) current; kinetic/ham PHYSICALLY DIRTY (RK 08:51)
        @test _pp_cur(plan, ow, sh, _pp_c(plan, :mom)) && _pp_cur(plan, ow, sh, _pp_c(plan, :pos))
        @test _pp_cur(plan, ow, sh, _pp_c(plan, :dpot_dpos)) && _pp_cur(plan, ow, sh, _pp_c(plan, :pot))
        @test !_pp_cur(plan, ow, sh, _pp_c(plan, :dkin_dmom))
        @test !_pp_cur(plan, ow, sh, _pp_c(plan, :kin)) && !_pp_cur(plan, ow, sh, _pp_c(plan, :ham))
        # NUMERICAL reversibility: leapfrog(h) then leapfrog(-h) returns to start
        lf(ow, sh, hs, (stepsize = -h,))
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :pos)) ≈ pos0
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :mom)) ≈ mom0
        # function-barrier warmed exact 0-B + @inferred
        cp2 = _PPFix.CountPgrad(0); o2, s2 = _pp_construct(pf, plan, _pp_values(plan, T, d, cp2))
        RK.compile_prepared_initialization(pf, typeof(o2), typeof(s2))(o2, s2, hs)
        lf2 = RK.compile_leapfrog(pf, typeof(o2), typeof(s2), leaf)
        kw = (stepsize = h,); lf2(o2, s2, hs, kw); lf2(o2, s2, hs, kw)
        @test (@allocated lf2(o2, s2, hs, kw)) == 0
        Test.@inferred lf2(o2, s2, hs, kw)
    end
    # CONTROLLED energy-error behavior: |Δham| stays small over a short trajectory (the HMC gate owns the
    # stronger epsilon-scaling law). Recompute ham via the full pass each step.
    d = 3; cp = _PPFix.CountPgrad(0); ow, sh = _pp_construct(pf, plan, _pp_values(plan, Float64, d, cp))
    init = RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh)); init(ow, sh, hs)
    lf = RK.compile_leapfrog(pf, typeof(ow), typeof(sh), leaf)
    ham0 = _pp_rd(plan, ow, sh, _pp_c(plan, :ham))
    for _ in 1:20; lf(ow, sh, hs, (stepsize = 0.05,)); init(ow, sh, hs); end
    @test abs(_pp_rd(plan, ow, sh, _pp_c(plan, :ham)) - ham0) < 0.01
end

@testset "executable leapfrog — stale-at-entry recovery: forced pgrad failure, RETRY analytic + grad Δ==2 (RK 09:06/09:08)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf)
    leaf = RK.method_irs(_PPFix.leapfrog!)[1]; d = 3; h = 0.1
    tg = _PPFix.ThrowOnGrad(0, 2)                                # ok at init (call 1); throws on the post-drift grad
    ow, sh = _pp_construct(pf, plan, _pp_values(plan, Float64, d, tg))
    RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs)     # grad call #1 (init)
    lf = RK.compile_leapfrog(pf, typeof(ow), typeof(sh), leaf)
    @test_throws ErrorException lf(ow, sh, hs, (stepsize = h,))                    # post-drift grad #2 THROWS
    # executed prefix committed (kick-1 mom + drift pos blessed); the throwing grad left dpot DIRTY
    pos_pfx = copy(_pp_rd(plan, ow, sh, _pp_c(plan, :pos))); mom_pfx = copy(_pp_rd(plan, ow, sh, _pp_c(plan, :mom)))
    @test _pp_cur(plan, ow, sh, _pp_c(plan, :mom)) && _pp_cur(plan, ow, sh, _pp_c(plan, :pos))
    @test !_pp_cur(plan, ow, sh, _pp_c(plan, :dpot_dpos))
    # RETRY: entry dpot is dirty → kick-1 entry-ensure recomputes it (grad), then post-drift grad → Δ == 2
    tg.throw_at = 999; n_pre = tg.n
    lf(ow, sh, hs, (stepsize = h,))
    @test tg.n - n_pre == 2                                      # entry dpot recovery + post-drift recompute
    # exact ANALYTIC recovery state (one leapfrog step from the committed prefix, gradient repaired at entry)
    M = Matrix{Float64}(I(d)) .+ 0.5
    mom_hh = mom_pfx .- h .* pos_pfx; pos_a = pos_pfx .+ h .* (M \ mom_hh); mom_a = mom_hh .- h .* pos_a
    @test _pp_rd(plan, ow, sh, _pp_c(plan, :pos)) ≈ pos_a
    @test _pp_rd(plan, ow, sh, _pp_c(plan, :mom)) ≈ mom_a
    # final masks correct: gradient current, kinetic/ham dirty
    @test _pp_cur(plan, ow, sh, _pp_c(plan, :dpot_dpos)) && _pp_cur(plan, ow, sh, _pp_c(plan, :pot))
    @test !_pp_cur(plan, ow, sh, _pp_c(plan, :dkin_dmom)) && !_pp_cur(plan, ow, sh, _pp_c(plan, :ham))
end

@testset "executable leapfrog — authored qualified-slot alias rebind is REJECTED (RK 08:55/08:59)" begin
    # a synthetic QUALIFIED mutable alias `Ops.oftype` (Ops initially Base, then a module with a different
    # oftype). No Base mutation — only the module-local alias binding moves. The def-time snapshot check must
    # reject the emission once the authored qualifier resolves away from the captured registration.
    modx = Module(:LFQ)
    Core.eval(modx, :(using ReactiveKernels, LinearAlgebra))
    Core.eval(modx, :(Ops = Base))
    Core.eval(modx, :(@kernel leapfrog!(phasepoint; stepsize) = begin
        @. phasepoint.mom -= Ops.oftype(stepsize, 0.5) * stepsize * phasepoint.dham_dpos
        @. phasepoint.pos +=                            stepsize * phasepoint.dham_dmom
        @. phasepoint.mom -= Ops.oftype(stepsize, 0.5) * stepsize * phasepoint.dham_dpos
    end))
    leafq = RK.method_irs(modx.leapfrog!)[1]
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); d = 3
    ow, sh = _pp_construct(pf, plan, _pp_values(plan, Float64, d, _PPFix.CountPgrad(0)))
    @test RK.compile_leapfrog(pf, typeof(ow), typeof(sh), leafq) isa Function      # BEFORE rebind: fine
    Core.eval(modx, :(module Evil; oftype(a, b) = error("evil"); end))
    Core.eval(modx, :(Ops = Evil))                                                 # move the authored qualifier
    @test_throws RK._LLowerReject RK.compile_leapfrog(pf, typeof(ow), typeof(sh), leafq)
end

@testset "executable leapfrog — a DIRTY non-producible source is REJECTED before any write/pgrad (RK 09:14)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf)
    leaf = RK.method_irs(_PPFix.leapfrog!)[1]; d = 3
    cp = _PPFix.CountPgrad(0); ow, sh = _pp_construct(pf, plan, _pp_values(plan, Float64, d, cp))
    RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs)     # VALID init
    lf = RK.compile_leapfrog(pf, typeof(ow), typeof(sh), leaf)
    pos_b = copy(_pp_rd(plan, ow, sh, _pp_c(plan, :pos))); mom_b = copy(_pp_rd(plan, ow, sh, _pp_c(plan, :mom)))
    n_b = cp.n
    # explicitly KILL the owned `mom` non-producible source mask (kick-1 reads it FIRST, authored order)
    fsm = RK.kernel_plan_field(plan, _pp_c(plan, :mom)); RK._canon_kill!(ow, Val(fsm[2]))
    @test_throws ErrorException lf(ow, sh, hs, (stepsize = 0.1,))   # dirty-source assert throws — cannot repair
    # thrown BEFORE any physical write / pgrad: state + grad-count unchanged
    @test cp.n == n_b                                              # NO pgrad ran
    @test _pp_rd(plan, ow, sh, _pp_c(plan, :pos)) == pos_b         # NO physical position write
    @test _pp_rd(plan, ow, sh, _pp_c(plan, :mom)) == mom_b         # NO physical momentum write
    @test !_pp_cur(plan, ow, sh, _pp_c(plan, :mom))                # source stays dirty (requires reset)
end
