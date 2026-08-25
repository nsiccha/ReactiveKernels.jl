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
