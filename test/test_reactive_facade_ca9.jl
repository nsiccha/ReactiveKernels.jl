using ReactiveKernels
using Test

# Exact ca9-shaped syntax fixture for the @reactive facade: indexed / nested
# property+index LHS, scalar and indexed compound, destructuring swap, dotted and
# @. broadcast mutation, loops, return assignment, method kwargs/defaults/sibling
# calls (including explicit __self__), local alias-root mutation with a
# materialized downstream dependent, typed constructor dispatch, cross-instance
# copyto!, exception-safe invalidation on a partial-write-then-throw, RHS-before-
# receiver order, and 0-alloc root mutation. Mirrors ReactiveHMC ca9.

const _CA9_TOTAL = Ref(0)   # recompute counter for `total`  (depends on scalar)
const _CA9_WSUM = Ref(0)    # recompute counter for `wsum`   (depends on weights)
const _CA9_MSUM = Ref(0)    # recompute counter for `msum`   (depends on mvs)

@reactive treestore(dim::Int, n::Int) = begin
    weights::Vector{Vector{Float64}} = [zeros(2) for _ in 1:n]
    slots::Vector{Vector{Float64}} = [fill(Float64(i), dim) for i in 1:n]
    scalar::Float64 = 0.0
    total::Float64 = (_CA9_TOTAL[] += 1; scalar * 2)
    wsum::Float64 = (_CA9_WSUM[] += 1; sum(sum, weights))          # DEPENDS on weights
    mvs::Vector{NamedTuple{(:mom,),Tuple{Vector{Float64}}}} =
        [(mom = fill(2.0, dim),) for _ in 1:n]                     # nested (ca9 trees.bwd.mom)
    msum::Float64 = (_CA9_MSUM[] += 1; sum(t -> sum(t.mom), mvs))  # DEPENDS on mvs
    setweight!(i, a, b) = begin
        weights[i][1] = a
        weights[i][2] = weights[i][1] + b
    end
    accumulate!(i, x) = begin
        weights[i][1] += x
    end
    bumpscalar!(x) = begin
        scalar += x
    end
    swapslots!(i, j) = begin
        slots[i], slots[j] = slots[j], slots[i]
    end
    scaleslot!(i, c) = begin
        slots[i] .*= c
    end
    fillslot!(i, v) = begin
        @. slots[i] = v
    end
    aliasbump!(depth, delta) = begin
        w = weights[depth]              # LOCAL alias into a field element
        w[1] += delta                   # mutate through the alias
    end
    aliasdot!(depth) = begin
        tr = mvs[depth]                 # LOCAL alias into a nested field element
        @. tr.mom = -slots[depth]       # RHS reads a DISTINCT reactive field (ca9: -bwd.mom)
    end
    partialthrow!() = begin
        # a single assign! closure that writes weights[1][2] then throws in the RHS
        weights[1][1] += (weights[1][2] = 9.0; error("boom"))
    end
    loopset!() = begin
        for k in 1:n
            weights[k][2] = scalar
        end
    end
    sameroot!(c, delta) = begin
        if c                            # both branches alias the SAME field `weights`
            w = weights[1]
        else
            w = weights[2]
        end
        w[1] += delta                   # sound single-root alias -> accepted, invalidates weights
    end
    ternroot!(c, delta) = begin
        w = c ? weights[1] : weights[2] # ternary RHS, same root `weights` -> accepted
        w[1] += delta
    end
    andsameroot!(c, delta) = begin
        w = weights[1]
        c && (w = weights[2])           # short-circuit rebind, SAME root -> accepted
        w[1] += delta
    end
    orsameroot!(c, delta) = begin
        w = weights[1]
        c || (w = weights[2])           # || short-circuit rebind, SAME root -> accepted
        w[1] += delta
    end
end

@testset "@reactive ca9 — indexed/nested/compound/dotted/@./loop" begin
    _CA9_TOTAL[] = 0
    t = treestore(3, 4)
    @test t.weights[1] == [0.0, 0.0]
    @test t.slots[2] == [2.0, 2.0, 2.0]
    @test t.total == 0.0 && _CA9_TOTAL[] == 1

    setweight!(t, 2, 5.0, 1.0)
    @test t.weights[2] == [5.0, 6.0]

    accumulate!(t, 2, 10.0)                 # weights[2][1] += 10 -> 15
    @test t.weights[2][1] == 15.0

    bumpscalar!(t, 3.0)                      # scalar override -> total invalidated
    @test t.scalar == 3.0 && t.total == 6.0 && _CA9_TOTAL[] == 2

    a2 = copy(t.slots[2]); a3 = copy(t.slots[3])
    swapslots!(t, 2, 3)
    @test t.slots[2] == a3 && t.slots[3] == a2

    scaleslot!(t, 1, 2.0)
    @test t.slots[1] == [2.0, 2.0, 2.0]
    fillslot!(t, 4, 7.0)
    @test t.slots[4] == [7.0, 7.0, 7.0]

    bumpscalar!(t, 4.0)                      # scalar = 7
    loopset!(t)
    @test all(w -> w[2] == 7.0, t.weights)
end

@testset "@reactive ca9 — alias-root mutation invalidates a downstream dependent" begin
    _CA9_WSUM[] = 0
    t = treestore(2, 3)
    @test t.wsum == 0.0 && _CA9_WSUM[] == 1
    setweight!(t, 1, 4.0, 0.0)              # weights[1] = [4, 4]
    @test t.wsum == 8.0 && _CA9_WSUM[] == 2 # direct field mutation recomputes wsum

    before = t.wsum; c = _CA9_WSUM[]
    aliasbump!(t, 1, 100.0)                 # w = weights[1]; w[1] += 100  (alias mutation)
    @test t.weights[1][1] == 104.0
    @test t.wsum == before + 100.0          # dependent recomputed via the alias path
    @test _CA9_WSUM[] == c + 1              # recomputation actually happened
end

@testset "@reactive ca9 — same-root if/else alias accepted (not over-rejected)" begin
    _CA9_WSUM[] = 0
    t = treestore(2, 3)
    @test t.wsum == 0.0 && _CA9_WSUM[] == 1        # materialize (all weights zero)
    c = _CA9_WSUM[]
    sameroot!(t, true, 5.0)                        # c=true -> w=weights[1]; weights[1][1]+=5
    @test t.weights[1][1] == 5.0
    @test t.wsum == 5.0 && _CA9_WSUM[] == c + 1    # weights invalidated -> wsum recomputed
    c2 = _CA9_WSUM[]
    sameroot!(t, false, 3.0)                       # c=false -> w=weights[2]; weights[2][1]+=3
    @test t.weights[2][1] == 3.0
    @test t.wsum == 8.0 && _CA9_WSUM[] == c2 + 1
    # ternary RHS with the SAME root is also accepted and invalidates the dependent
    c3 = _CA9_WSUM[]
    ternroot!(t, true, 10.0)                       # w = true ? weights[1] : weights[2]
    @test t.weights[1][1] == 15.0                  # 5 + 10
    @test t.wsum == 18.0 && _CA9_WSUM[] == c3 + 1
    # short-circuit && rebind to the SAME root is accepted; prove BOTH paths
    c4 = _CA9_WSUM[]
    andsameroot!(t, false, 2.0)                    # c=false -> no rebind; w=weights[1]; +2
    @test t.weights[1][1] == 17.0
    @test t.wsum == 20.0 && _CA9_WSUM[] == c4 + 1
    c5 = _CA9_WSUM[]
    andsameroot!(t, true, 4.0)                     # c=true -> executed RHS; w=weights[2] (=3) +4
    @test t.weights[2][1] == 7.0
    @test t.wsum == 24.0 && _CA9_WSUM[] == c5 + 1
    # same-root `||` control, both paths
    c6 = _CA9_WSUM[]
    orsameroot!(t, true, 1.0)                      # c=true -> short-circuit; w=weights[1]; +1
    @test t.weights[1][1] == 18.0
    @test t.wsum == 25.0 && _CA9_WSUM[] == c6 + 1
    c7 = _CA9_WSUM[]
    orsameroot!(t, false, 6.0)                     # c=false -> executed RHS; w=weights[2] (=7) +6
    @test t.weights[2][1] == 13.0
    @test t.wsum == 31.0 && _CA9_WSUM[] == c7 + 1
    # accepted short-circuit aliases must stay type-stable + 0-alloc on BOTH paths
    # (the conditionally-rebound alias local is captured via a fresh single binding,
    # not boxed) — matching the ternary's Float64 / 0 B.
    _and_ret(o, c, x) = andsameroot!(o, c, x)
    _or_ret(o, c, x) = orsameroot!(o, c, x)
    @test @inferred(_and_ret(t, false, 0.0)) isa Float64
    @test @inferred(_and_ret(t, true, 0.0)) isa Float64
    @test @inferred(_or_ret(t, false, 0.0)) isa Float64
    @test @inferred(_or_ret(t, true, 0.0)) isa Float64
    _aalloc(o, c, x) = (andsameroot!(o, c, x); @allocated andsameroot!(o, c, x))
    _oalloc(o, c, x) = (orsameroot!(o, c, x); @allocated orsameroot!(o, c, x))
    @test _aalloc(t, false, 0.0) == 0 && _aalloc(t, true, 0.0) == 0
    @test _oalloc(t, false, 0.0) == 0 && _oalloc(t, true, 0.0) == 0
end

@testset "@reactive ca9 — local-alias @. nested mutation invalidates dependent" begin
    _CA9_MSUM[] = 0
    t = treestore(2, 3)
    @test t.msum == 12.0 && _CA9_MSUM[] == 1     # 3 * sum([2,2]) = 12
    c = _CA9_MSUM[]
    aliasdot!(t, 1)                              # tr = mvs[1]; @. tr.mom = -slots[1] (=[1,1])
    @test t.mvs[1].mom == [-1.0, -1.0]           # nested @. mutation, RHS a distinct field
    @test t.msum == 6.0                          # -2 (mvs[1]) + 4 + 4 = 6, recomputed
    @test _CA9_MSUM[] == c + 1                    # dependent recomputed via the alias path
end

@testset "@reactive ca9 — partial-write-then-throw invalidates same-root dependent" begin
    _CA9_WSUM[] = 0
    t = treestore(2, 2)
    @test t.wsum == 0.0
    c = _CA9_WSUM[]
    @test_throws ErrorException partialthrow!(t)   # writes weights[1][2]=9 then throws
    @test t.weights[1][2] == 9.0                    # the partial write landed
    @test t.wsum == 9.0                             # assign!'s finally invalidated wsum
    @test _CA9_WSUM[] == c + 1                       # ...and it recomputed to include the 9
end

@reactive sampler_like(x::Float64) = begin
    y::Float64 = x + 1.0
    flag::Bool = true
    may_sample::Bool = true
    setflag!(; force::Bool = false) = begin
        flag = force ? true : !flag
    end
    advance!(i::Int, j::Int = 2) = begin
        setflag!(; force = i > j)
        setflag!(__self__)
    end
    stop!() = begin
        may_sample || return may_sample = false
        return may_sample = false
    end
    setY!(a) = begin
        return y = a * 2.0
    end
end

@testset "@reactive ca9 — method kwargs/defaults/sibling/__self__/return/typed" begin
    s = sampler_like(2.0)
    @test s.y == 3.0 && s.flag === true && s.may_sample === true

    setflag!(s)                               # force=false -> !true = false
    @test s.flag === false
    setflag!(s; force = true)
    @test s.flag === true

    advance!(s, 3, 2)                         # force=true -> true; then self -> !true=false
    @test s.flag === false
    advance!(s, 1)                            # j=2; force=false -> !false=true; self -> false
    @test s.flag === false

    @test setY!(s, 5.0) == 10.0               # return assignment returns the value
    @test s.y == 10.0
    @test stop!(s) === false                  # return may_sample = false -> returns false
    @test s.may_sample === false

    @test !applicable(sampler_like, 1)        # typed ctor dispatch preserved (Int !~ Float64)
    @test applicable(sampler_like, 1.0)
end

@reactive orderobj(k::Int) = begin
    base::Vector{Float64} = fill(1.0, k)
    derived::Vector{Float64} = base .+ 0.0
    poke!(i) = begin
        derived[i] = (base[1] = 41.0; base[i] + 1.0)
    end
end

@testset "@reactive ca9 — cross-instance copyto!, RHS-first order, 0-alloc" begin
    # independently constructed instances share the program -> copyto!-compatible
    a = treestore(2, 3); b = treestore(2, 3)
    setweight!(a, 1, 9.0, 0.0)
    copyto!(b, a)
    @test b.weights[1] == a.weights[1]
    @test a.program === b.program

    # plain `=` RHS-before-receiver order: RHS mutates an upstream field of the
    # root (base), invalidating it, before the derived receiver is materialized
    o = orderobj(3)
    poke!(o, 1)
    @test o.base[1] == 41.0
    @test o.derived[1] == 42.0                # 41 (new base[1]) + 1, into recomputed derived

    # 0-alloc root mutation behind a function barrier (after warmup)
    z = treestore(4, 3)
    _acc(obj, i, x) = (accumulate!(obj, i, x); @allocated accumulate!(obj, i, x))
    @test _acc(z, 1, 1.0) == 0
end

const _BR_A = Ref(0)
const _BR_B = Ref(0)

@reactive branchobj(k::Int) = begin
    a::Vector{Float64} = fill(1.0, k)
    b::Vector{Float64} = fill(10.0, k)
    asum::Float64 = (_BR_A[] += 1; sum(a))
    bsum::Float64 = (_BR_B[] += 1; sum(b))
    bumpa!(delta) = begin
        w = a                          # straight-line single-field alias (ca9 shape)
        w[1] += delta
    end
end

# Expand a @reactive definition, returning "" on success or the ArgumentError
# message on a macro-expansion rejection.
function _reactive_expansion_error(defexpr)
    try
        macroexpand(@__MODULE__, defexpr)
        ""
    catch e
        e isa LoadError && (e = e.error)
        e isa ArgumentError ? e.msg : sprint(showerror, e)
    end
end

@testset "@reactive ca9 — single-field alias is scoped; frozen other field untouched" begin
    _BR_A[] = 0; _BR_B[] = 0
    t = branchobj(2)
    @test t.asum == 2.0 && t.bsum == 20.0                 # materialize both dependents
    ReactiveKernels.freeze!(t.state, t.handles.b)         # freeze the b field
    ca = _BR_A[]; cb = _BR_B[]
    bumpa!(t, 5.0)                                        # w = a; a[1] 1 -> 6 (no throw)
    @test t.a[1] == 6.0 && t.b[1] == 10.0                 # only a mutated
    @test t.asum == 7.0 && _BR_A[] == ca + 1              # a's dependent recomputed
    @test t.bsum == 20.0 && _BR_B[] == cb                 # b frozen: NOT materialized/recomputed
end

@testset "@reactive ca9 — control-flow-divergent aliases are rejected at expansion" begin
    # different fields on different branches
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_branch(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k); b::Vector{Float64} = fill(2.0, k)
            m!(c) = begin
                if c; w = a; else; w = b; end
                w[1] += 1.0
            end
        end)))
    # alias on one path, non-alias on the other (bare if fall-through)
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_mixed(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k)
            m!(c) = begin
                w = a
                if c; w = zeros(k); end
                w[1] += 1.0
            end
        end)))
    # loop rebinds the alias to a different field (0 vs >=1 iterations diverge)
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_loop(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k); b::Vector{Float64} = fill(2.0, k)
            m!(n) = begin
                w = a
                for _ in 1:n; w = b; end
                w[1] += 1.0
            end
        end)))
    # ternary RHS aliasing DIFFERENT fields
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_tern(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k); b::Vector{Float64} = fill(2.0, k)
            m!(c) = begin
                w = c ? a : b
                w[1] += 1.0
            end
        end)))
    # ternary RHS aliasing a field on one side, a non-alias on the other
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_tern_mixed(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k)
            m!(c) = begin
                w = c ? a : zeros(k)
                w[1] += 1.0
            end
        end)))
    # short-circuit && rebinding the alias to a DIFFERENT field
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_and(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k); b::Vector{Float64} = fill(2.0, k)
            m!(c) = begin
                w = b
                c && (w = a)
                w[1] += 1.0
            end
        end)))
    # short-circuit || rebinding the alias to a DIFFERENT field
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_or(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k); b::Vector{Float64} = fill(2.0, k)
            m!(c) = begin
                w = b
                c || (w = a)
                w[1] += 1.0
            end
        end)))
    # short-circuit RHS `w = c && a` (alias-vs-nonalias)
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_andrhs(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k)
            m!(c) = begin
                w = c && a
                w[1] += 1.0
            end
        end)))
    # short-circuit RHS `w = c || a`
    @test occursin("cannot be soundly invalidated", _reactive_expansion_error(:(
        @reactive _bad_orrhs(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k)
            m!(c) = begin
                w = c || a
                w[1] += 1.0
            end
        end)))
    # a straight-line single-field alias must NOT be rejected
    @test _reactive_expansion_error(:(
        @reactive _ok_alias(k::Int) = begin
            a::Vector{Float64} = fill(1.0, k)
            m!() = begin
                w = a
                w[1] += 1.0
            end
        end)) == ""
end

@reactive retobj(k::Int) = begin
    arr::Vector{Float64} = fill(2.0, k)
    scal::Float64 = 0.0
    r_compound!(i) = begin
        return arr[i] += 1.0           # indexed compound -> new scalar
    end
    r_dotted!(v) = begin
        return arr .= v                # dotted -> destination (arr)
    end
    r_scalarcompound!(x) = begin
        return scal += x               # scalar bare-field compound -> new value
    end
    r_wholefield!(v) = begin
        return arr = v                 # whole-field -> v
    end
    r_alias!(i) = begin
        w = arr
        return w[i] += 1.0             # local-alias compound -> new scalar
    end
    r_dotmacro!(v) = begin
        return @. arr = v              # @. dotmacro -> destination (arr)
    end
    r_aliasdotmacro!() = begin
        w = arr
        return @. w = w * 2.0          # alias-@. -> destination (w === arr)
    end
end

@testset "@reactive ca9 — rooted in-place assignment return values" begin
    o = retobj(3)
    @test r_compound!(o, 1) === 3.0                # arr[1] 2 -> 3
    @test o.arr[1] == 3.0
    dest = r_dotted!(o, 5.0)                       # arr .= 5 returns the destination
    @test dest == [5.0, 5.0, 5.0] && dest === o.arr
    @test r_scalarcompound!(o, 4.0) === 4.0        # scal 0 -> 4
    @test r_wholefield!(o, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    @test r_alias!(o, 2) === 3.0                   # arr now [1,2,3]; w[2] 2 + 1 = 3
    d2 = r_dotmacro!(o, 7.0)                        # @. arr = 7 -> destination
    @test d2 == [7.0, 7.0, 7.0] && d2 === o.arr
    d3 = r_aliasdotmacro!(o)                        # w=arr; @. w = w*2 -> destination
    @test d3 == [14.0, 14.0, 14.0] && d3 === o.arr
end
