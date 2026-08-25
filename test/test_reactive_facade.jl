using ReactiveKernels
using Test

# Non-HMC exercise of the public @reactive object/method facade: signature args
# are mutable HAVE sources, body assignments are compiled reactive derived
# nodes, inner defs are ordinary methods taking the object first. No HMC here.

# Count recipe evaluations to prove reactive (minimal) recomputation.
const _FACADE_CALLS = (cost = Ref(0), shifted = Ref(0))

@reactive particle(pos::Vector{Float64}, scale::Float64) = begin
    cost::Float64 = (_FACADE_CALLS.cost[] += 1; scale * sum(abs2, pos))
    shifted::Vector{Float64} = (_FACADE_CALLS.shifted[] += 1; pos .+ 1.0)
    energy() = cost + sum(shifted)
    bump!(delta) = begin
        pos = pos .+ delta
    end
    rescale!(s) = begin
        scale = s
    end
    combo!(delta, s) = begin
        bump!(delta)
        rescale!(s)
    end
end

@testset "@reactive facade — invalidation, methods, hygiene, inference" begin
    _FACADE_CALLS.cost[] = 0; _FACADE_CALLS.shifted[] = 0
    p = particle([1.0, 2.0], 2.0)
    @test p isa ReactiveObject

    # derived reactive nodes compute lazily, once
    @test p.cost == 2.0 * (1.0 + 4.0)
    @test p.shifted == [2.0, 3.0]
    @test (_FACADE_CALLS.cost[], _FACADE_CALLS.shifted[]) == (1, 1)
    @test p.cost == 10.0                      # cached, no recompute
    @test _FACADE_CALLS.cost[] == 1

    # method mutates a HAVE source -> downstream derived nodes invalidate
    bump!(p, [1.0, 1.0])
    @test p.pos == [2.0, 3.0]
    @test p.cost == 2.0 * (4.0 + 9.0)         # recomputed
    @test p.shifted == [3.0, 4.0]
    @test _FACADE_CALLS.cost[] == 2

    # mutating one source recomputes only its dependents (both depend on pos here,
    # scale-only change leaves `shifted` valid)
    s0 = _FACADE_CALLS.shifted[]
    rescale!(p, 3.0)
    @test p.cost == 3.0 * (4.0 + 9.0)
    @test _FACADE_CALLS.shifted[] == s0        # shifted independent of scale

    # a method reading derived fields
    @test energy(p) == p.cost + sum(p.shifted)

    # a method calling sibling methods (forwarded with the object)
    combo!(p, [1.0, 0.0], 1.0)
    @test p.pos == [3.0, 3.0]
    @test p.cost == 1.0 * (9.0 + 9.0)

    # type inference through the object property getter and a method
    _cost(o) = o.cost
    @test @inferred(_cost(p)) == p.cost
    @test @inferred(energy(p)) == energy(p)

    # hygiene: caller globals with field names do not leak into the object
    pos = 999.0; scale = -1.0; cost = :nope   # shadow the field names in caller scope
    q = particle([5.0], 4.0)
    @test q.cost == 4.0 * 25.0
    @test q.pos == [5.0]

    # independence / composition: two objects share no state; copy is detached
    a = particle([1.0], 1.0); b = particle([1.0], 1.0)
    bump!(a, [10.0])
    @test a.pos == [11.0] && b.pos == [1.0]
    c = copy(a)
    bump!(a, [100.0])
    @test a.pos == [111.0] && c.pos == [11.0]
    @test c.program === a.program              # program shared, state detached
end

@reactive kwparticle(pos::Vector{Float64}; scale::Float64 = 2.0, shift::Float64 = 0.0) = begin
    cost::Float64 = scale * sum(abs2, pos) + shift
    tweak!(s) = begin
        scale = s
    end
end

@testset "@reactive facade — keyword/default sources" begin
    p = kwparticle([1.0, 2.0])                    # defaults scale=2, shift=0
    @test p.cost == 2.0 * 5.0
    @test p.scale == 2.0 && p.shift == 0.0
    q = kwparticle([1.0, 2.0]; scale = 3.0, shift = 1.0)
    @test q.cost == 3.0 * 5.0 + 1.0
    tweak!(q, 10.0)                               # mutate a keyword-source field
    @test q.cost == 10.0 * 5.0 + 1.0
    kp = kwparticle([1.0]; scale = 5.0)
    _kwcost(o) = o.cost
    @test @inferred(_kwcost(kp)) == 5.0           # type-stable read behind a barrier
end

# Derived-field override (assign! lowering) + method keyword/default + sibling
# forwarding, exactly the ca9 shapes (gofwd/may_continue writes; step!(;force),
# swapproposal!(i, j=...)).
@reactive counter(base::Int) = begin
    derived::Int = base + 10
    total::Int = derived * 2
    override!(v::Int) = begin
        derived = v                       # assign a DERIVED field: temporary override
    end
    bumpbase!(; by::Int = 1) = begin      # keyword-only method
        base = base + by
    end
    combo!(v::Int, n::Int = 2) = begin    # positional default + sibling calls
        override!(v)                      # sibling call (positional)
        bumpbase!(; by = n)               # sibling call with a keyword
    end
end

@testset "@reactive facade — derived override + method kwargs/defaults" begin
    c = counter(5)
    @test c.derived == 15 && c.total == 30

    # a method overrides a DERIVED field: get! returns the override (no recompute),
    # and its downstream recomputes from the override
    override!(c, 100)
    @test c.derived == 100
    @test c.total == 200

    # a later true-upstream change invalidates the override -> recompute from recipe
    bumpbase!(c; by = 3)
    @test c.base == 8
    @test c.derived == 18                 # 8 + 10, override discarded
    @test c.total == 36

    # keyword default
    bumpbase!(c)                          # by defaults to 1
    @test c.base == 9 && c.derived == 19

    # positional default + sibling forwarding (positional and keyword)
    combo!(c, 999, 4)                     # override(999) then bumpbase!(by=4)
    @test c.base == 13                    # 9 + 4
    @test c.derived == 23                 # override 999 superseded by base change
    combo!(c, 0)                          # n defaults to 2
    @test c.base == 15                    # 13 + 2
end
