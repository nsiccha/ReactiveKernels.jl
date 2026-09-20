# Merge tests: `Base.merge` splices override statements into a copy of a base
# `RKPPLModel` (StanBlocks-style program composition over shared blocks).
# Every merge is transparency-checked the submodel way: the merged model
# lowers to the identical `StructuralPlan` as the hand-written variant.
# Helpers (_plans_equal, _gen_columns, _query, _check_gradient) come from the
# earlier includes; this file defines its own hand-shared blocks.

@rkppl merge_latent_normal(s) = begin
    r ~ Normal(0, s)
    r
end

@rkppl merge_normal_stream(eta, sigma) = begin
    slot .~ Normal.(eta, sigma)
    slot
end

@rkppl merge_normal_stream_tight(eta) = begin
    u ~ Exponential(2.0)
    slot .~ Normal.(eta, u)
    slot
end

# Kulprit_m0-shaped gaussian GLM on the hand-shared blocks.
merge_base = @rkppl begin
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    s ~ merge_latent_normal(1.0)
    mu = a .+ b .* x
    y ~ merge_normal_stream(mu, s)
end

# Same shape with an explicit scale (family-swap target).
merge_scale_base = @rkppl begin
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    s ~ Exponential(1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
end

_merge_lower(m) = lower_rkppl(m.ast, (:y, :x); mod = @__MODULE__)
_merge_s_family(params) = first(p.family for p in params if p.name === :s)

@testset "merge prior-swap transparency" begin
    got = _merge_lower(Base.merge(merge_base, :(a ~ Normal(0, 10))))
    hand = @rkppl begin
        a ~ Normal(0, 10)
        b ~ Normal(0, 2)
        s ~ merge_latent_normal(1.0)
        mu = a .+ b .* x
        y ~ merge_normal_stream(mu, s)
    end
    @test _plans_equal(got, _merge_lower(hand))
    # Role-carry: the swapped name keeps its Intercept coefficient row,
    # with the new scale.
    row = first(p for p in got.population_priors if p.addressee === :Intercept)
    @test (row.predictor, row.location, row.scale) === (:mu, 0.0, 10.0)
end

@testset "merge family-swap transparency" begin
    got = _merge_lower(Base.merge(merge_scale_base, :(s ~ HalfNormal(1.0))))
    hand = @rkppl begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        s ~ HalfNormal(1.0)
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end
    @test _plans_equal(got, _merge_lower(hand))
    # The swap is observable on the sampled scale's family.
    @test _merge_s_family(got.parameters) !==
        _merge_s_family(_merge_lower(merge_scale_base).parameters)
end

@testset "merge use-site swap transparency" begin
    got = _merge_lower(Base.merge(merge_base,
        :(y ~ merge_normal_stream_tight(mu))))
    hand = @rkppl begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        s ~ merge_latent_normal(1.0)
        mu = a .+ b .* x
        y ~ merge_normal_stream_tight(mu)
    end
    @test _plans_equal(got, _merge_lower(hand))
    # The swap is observable: the tight stream brings its namespaced scale.
    @test !_plans_equal(got, _merge_lower(merge_base))
    @test any(p -> p.name === :y_u, got.parameters)
    @test length(got.responses) == 1 && got.responses[1].response === :y
end

@testset "merge base unchanged" begin
    before_ast = deepcopy(merge_base.ast)
    before_fixed = deepcopy(merge_base.fixed)
    merged = Base.merge(merge_base, :(a ~ Normal(0, 10)))
    @test merged !== merge_base
    @test merge_base.ast == before_ast
    @test merged.ast != merge_base.ast
    @test merge_base.fixed == before_fixed
    @test merged.fixed == before_fixed
    fixed = Base.merge(merge_base, (; s = fill(2.0, 6)))
    @test fixed.fixed[:s] == fill(2.0, 6)
    @test fixed.fixed !== merge_base.fixed
    @test merge_base.fixed == before_fixed
end

@testset "merge completion appends fresh LHS" begin
    stripped = @rkppl begin
        a ~ Normal(0, 5)
        s ~ merge_latent_normal(1.0)
        mu = a .+ b .* x
        y ~ merge_normal_stream(mu, s)
    end
    got = _merge_lower(Base.merge(stripped, :(b ~ Normal(0, 2))))
    hand = @rkppl begin
        a ~ Normal(0, 5)
        s ~ merge_latent_normal(1.0)
        mu = a .+ b .* x
        y ~ merge_normal_stream(mu, s)
        b ~ Normal(0, 2)
    end
    @test _plans_equal(got, _merge_lower(hand))
end

@testset "merge NamedTuple fix transparency" begin
    n = 6
    fixed = Base.merge(merge_base, (; s = fill(2.0, n)))
    hand = @rkppl begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y ~ merge_normal_stream(mu, s)
    end
    cols, _ = _gen_columns()
    @test length(cols[:y]) == n
    # The fixed value travels: no `s` kwarg needed at the call.
    bf = fixed(; y = cols[:y], x = cols[:x])
    @test bf.columns[:s] == fill(2.0, n)
    bh = hand(; y = cols[:y], x = cols[:x], s = fill(2.0, n))
    @test _plans_equal(bf, bh)
    # Explicit kwargs win over fixed (SB easily-rebound data).
    rebound = fixed(; y = cols[:y], x = cols[:x], s = ones(n))
    @test rebound.columns[:s] == ones(n)
    # Fix-wins over a splice naming the same LHS in one call.
    both = Base.merge(merge_base, :(s ~ merge_latent_normal(2.0)),
        (; s = fill(2.0, n)))
    @test both.fixed[:s] == fill(2.0, n)
    bb = both(; y = cols[:y], x = cols[:x])
    @test _plans_equal(bb, bh)
end

@testset "merge chaining" begin
    chained = Base.merge(
        Base.merge(Base.merge(merge_base, :(a ~ Normal(0, 10))),
            :(b ~ Normal(0, 3))),
        :(s ~ merge_latent_normal(2.0)))
    single = Base.merge(merge_base, quote
        a ~ Normal(0, 10)
        b ~ Normal(0, 3)
        s ~ merge_latent_normal(2.0)
    end)
    hand = @rkppl begin
        a ~ Normal(0, 10)
        b ~ Normal(0, 3)
        s ~ merge_latent_normal(2.0)
        mu = a .+ b .* x
        y ~ merge_normal_stream(mu, s)
    end
    @test _plans_equal(_merge_lower(chained), _merge_lower(hand))
    @test _plans_equal(_merge_lower(single), _merge_lower(hand))
end

@testset "merge loud failures" begin
    # Not a statement.
    @test_throws SurfaceLoweringError Base.merge(merge_base, :(x + 1))
    @test_throws SurfaceLoweringError Base.merge(merge_base, :b)
    # Not a bare-Symbol LHS (indexed overrides are deferred).
    @test_throws SurfaceLoweringError Base.merge(merge_base, :(c[1] = 2))
    @test_throws SurfaceLoweringError Base.merge(merge_base,
        Expr(:call, :.~, :(y[1:3]), :(Normal.(mu, s))))
    # Nested blocks do not splice.
    @test_throws SurfaceLoweringError Base.merge(merge_base,
        quote
            begin
                a ~ Normal(0, 10)
            end
        end)
    # A broken base (duplicate LHS) fails closed, naming the name.
    dup = RKPPLModel(quote
        a ~ Normal(0, 1)
        a ~ Normal(0, 2)
        y .~ Normal.(a, 1.0)
    end, @__MODULE__)
    @test_throws SurfaceLoweringError Base.merge(dup, :(a ~ Normal(0, 1)))
    # Override naming a levels/ref stem: indexed overrides deferred.
    lev = @rkppl begin
        a ~ Normal(0, 1)
        c[levels(g)[2:end]] .~ Normal.(0, 2)
        mu = a .+ c[g]
        y .~ Normal.(mu, 1.5)
    end
    err = try
        Base.merge(lev, :(c ~ Normal(0, 1)))
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError
    # Fixing an unknown name or a non-vector value fails closed.
    @test_throws SurfaceLoweringError Base.merge(merge_base, (; nosuch = ones(6)))
    @test_throws SurfaceLoweringError Base.merge(merge_base, (; s = 2.0))
    @test_throws SurfaceLoweringError Base.merge(lev, (; c = ones(6)))
    # A plate-cell name is invisible to the top-level matcher: the append
    # collides at lowering through the single-assignment gate (still loud).
    plated = RKPPLModel(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(4),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:block, LineNumberNode(5),
                        :(t = a .+ b .* x[i]),
                        :(y[i] ~ Normal.(t, s)))))), @__MODULE__)
    shadowed = Base.merge(plated, :(t = a .+ b .* x))
    @test_throws SurfaceLoweringError lower_rkppl(shadowed.ast, (:y, :x))
end

@testset "merge joint session over shared defs" begin
    m1 = @rkppl begin
        a ~ Normal(0, 1)
        s ~ merge_latent_normal(1.0)
        mu = a .+ b .* x
        y ~ merge_normal_stream(mu, s)
    end
    m2 = @rkppl begin
        a ~ Normal(0, 3)
        t ~ merge_latent_normal(2.0)
        nu = a .+ b .* x
        y ~ merge_normal_stream(nu, t)
    end
    p1 = _merge_lower(m1)
    p2 = _merge_lower(m2)
    @test any(p -> p.name === :s_r, p1.parameters)
    @test !any(p -> p.name === :t_r, p1.parameters)
    @test any(p -> p.name === :t_r, p2.parameters)
    @test !any(p -> p.name === :s_r, p2.parameters)
end

@testset "merge demo end to end" begin
    cols, _ = _gen_columns()
    merged = Base.merge(merge_scale_base, :(s ~ HalfNormal(1.0)))
    hand = @rkppl begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        s ~ HalfNormal(1.0)
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end
    bm = merged(; y = cols[:y], x = cols[:x])
    bh = hand(; y = cols[:y], x = cols[:x])
    @test _plans_equal(bm, bh)
    u = [0.5, -0.25, 0.1]
    @test _query(build_kernel(bm).spec, bm, :posterior, u) ==
          _query(build_kernel(bh).spec, bh, :posterior, u)
    _check_gradient(build_kernel(bm).spec, bm, u)
    # The fix path queries identically to explicit data.
    n = length(cols[:y])
    fixed = Base.merge(merge_base, (; s = fill(2.0, n)))
    handf = @rkppl begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y ~ merge_normal_stream(mu, s)
    end
    bf = fixed(; y = cols[:y], x = cols[:x])
    bhf = handf(; y = cols[:y], x = cols[:x], s = fill(2.0, n))
    @test _plans_equal(bf, bhf)
    uf = [0.5, -0.25]
    @test _query(build_kernel(bf).spec, bf, :posterior, uf) ==
          _query(build_kernel(bhf).spec, bhf, :posterior, uf)
end
