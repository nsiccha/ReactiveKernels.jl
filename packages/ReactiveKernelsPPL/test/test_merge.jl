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

_merge_lower(m) = lower_rkppl(m.ast, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
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
    # The declared scalar owns its density and coordinates even as a coefficient.
    row = only(p for p in got.parameters if p.name === :a)
    @test (row.args.arg1, row.args.arg2) == (0, 10)
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
    @test any(p -> _test_scope_name(got, p.name) === :y_u, got.parameters)
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

# A statement that declares a fresh name appends whatever its shape, exactly as
# the hand-written model would carry it (snag rkppl-merge-cann-31516008).
@testset "merge appends fresh sized declarations and blocks" begin
    base = @rkppl begin
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1)
    end
    before = deepcopy(base.ast)
    _lower_y(m) = lower_rkppl(m.ast, (:y,); mod = @__MODULE__, conditioned = (:y,))
    Y = [0.3, -0.8, 1.1, 0.2]
    # A fresh sized vector, alone and with a definition reading it.
    sized = Base.merge(base, :(z[1:3] .~ Normal.(0, 1)))
    @test _plans_equal(_lower_y(sized), _lower_y(@rkppl begin
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1)
        z[1:3] .~ Normal.(0, 1)
    end))
    used = Base.merge(base, quote
        z[1:3] .~ Normal.(0, 1)
        y .~ Normal.(mu .+ sum(z), 1)
    end)
    hand_used = @rkppl begin
        mu ~ Normal(0, 1)
        y .~ Normal.(mu .+ sum(z), 1)
        z[1:3] .~ Normal.(0, 1)
    end
    @test _plans_equal(_lower_y(used), _lower_y(hand_used))
    bu, bh = used() | (; y = Y), hand_used() | (; y = Y)
    u = [0.1, -0.2, 0.4, 0.3]
    @test _query(build_kernel(bu).spec, bu, :posterior, u) ==
        _query(build_kernel(bh).spec, bh, :posterior, u)
    # A fresh indexed `~` and a fresh slice declaration append the same way.
    for decl in (:(b[1:2] ~ MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])),
                 :(eachrow(B[1:2, 1:2]) .~ MvNormal(zeros(2), [1.0 0.0; 0.0 1.0])))
        @test Base.merge(base, decl).ast.args[end] == decl
    end
    # A top-level `@scan` appends; its carried array is a model value.
    scanned = Base.merge(base, quote
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:3
                h[t] ~ Normal(0.5 * h[t - 1], 1)
            end
        end
        y .~ Normal.(mu .+ h[3], 1)
    end)
    hand_scanned = @rkppl begin
        mu ~ Normal(0, 1)
        y .~ Normal.(mu .+ h[3], 1)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:3
                h[t] ~ Normal(0.5 * h[t - 1], 1)
            end
        end
    end
    @test _plans_equal(_lower_y(scanned), _lower_y(hand_scanned))
    bs, bhs = scanned() | (; y = Y), hand_scanned() | (; y = Y)
    @test _query(build_kernel(bs).spec, bs, :posterior, u) ==
        _query(build_kernel(bhs).spec, bhs, :posterior, u)
    # A top-level `@plate` appends too.
    plated = Base.merge(base, quote
        @plate for i in eachindex(y)
            e[i] ~ Normal(0, 1)
        end
    end)
    @test plated.ast.args[end].args[1] === Symbol("@plate")
    @test _plans_equal(_lower_y(plated), _lower_y(@rkppl begin
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1)
        @plate for i in eachindex(y)
            e[i] ~ Normal(0, 1)
        end
    end))
    # The base is unchanged.
    @test base.ast == before
    # refused: an indexed assignment writes an element of an undeclared name (merge docstring; single assignment, §9).
    @test_throws SurfaceLoweringError Base.merge(base, :(c[1] = 2))
    # refused: an appended block rebinding a base name fails at lowering, as hand-written (single assignment, §9).
    clash = Base.merge(base, :(@scan begin
        mu[1] ~ Normal(0, 1)
        for t in 2:3
            mu[t] ~ Normal(mu[t - 1], 1)
        end
    end))
    @test_throws SurfaceLoweringError _lower_y(clash)
end

# A model text states a law (`c[levels(g)] .~ Normal.(0, sg)`), and an overlay
# passed to `merge` re-spells it non-centred without editing that text: the
# overlay's definition of `c` takes the declaration's place and its fresh
# innovation array appends (decision `1uqqq2o`). The result is the
# hand-written non-centred model. Its density is the centred model's at the
# mapped point plus the log-Jacobian of the map, one `log(scale)` per element.
# `scale` reads that scale off a constrained draw.
function _merge_respelled_checks(centred, overlay, hand, name::Symbol, scale, g, y)
    lower(m) = lower_rkppl(m.ast, (:y, :g); mod = @__MODULE__, conditioned = (:y,))
    value(q, u) = Base.invokelatest(q, u)
    before = deepcopy(centred.ast)
    respelled = Base.merge(centred, overlay)
    @test _plans_equal(lower(respelled), lower(hand))
    @test centred.ast == before
    pc, pn = centred(; g) | (; y), respelled(; g) | (; y)
    bc, bn = build_kernel(pc), build_kernel(pn)
    # The centred model samples the declared array. The re-spelled model
    # samples the innovations `z`, and the array is a value read by name.
    template = constrain(bc.layout, zeros(bc.layout.total))
    @test haskey(template, name)
    un = collect(range(-0.4, 0.5; length = bn.layout.total))
    drawn = constrain(bn.layout, un)
    @test haskey(drawn, :z) && !haskey(drawn, name)
    mapped = value(prepare_query(bn, pn, (name,)), un)
    @test length(mapped) == length(drawn.z) == length(template[name])
    shared = (; (k => drawn[k] for k in keys(drawn) if k !== :z && haskey(template, k))...)
    uc = unconstrain(bc.layout, merge(template, shared,
        NamedTuple{(name,)}((collect(Float64, mapped),))))
    @test value(prepare_query(bn, pn, :likelihood), un) ≈
        value(prepare_query(bc, pc, :likelihood), uc)
    @test value(prepare_query(bn, pn, :sampler), un) ≈
        value(prepare_query(bc, pc, :sampler), uc) + length(mapped) * log(scale(drawn))
end

@testset "merge re-spells a centred declaration as non-centred" begin
    g = ["a", "b", "b", "c", "a", "c", "b"]
    y = [0.3, -0.8, 1.1, 0.2, -0.4, 0.9, 0.5]
    @testset "a real-valued effect per level" begin
        centred = @rkppl begin
            a ~ Normal(0, 5)
            sg ~ HalfNormal(1)
            c[levels(g)] .~ Normal.(0, sg)
            sigma ~ Exponential(1)
            mu = a .+ c[g]
            y .~ Normal.(mu, sigma)
        end
        overlay = quote
            z[levels(g)] .~ Normal.(0, 1)
            c = sg .* z
        end
        hand = @rkppl begin
            a ~ Normal(0, 5)
            sg ~ HalfNormal(1)
            c = sg .* z
            sigma ~ Exponential(1)
            mu = a .+ c[g]
            y .~ Normal.(mu, sigma)
            z[levels(g)] .~ Normal.(0, 1)
        end
        _merge_respelled_checks(centred, overlay, hand, :c, d -> d.sg, g, y)
    end
    @testset "a location and a scale" begin
        centred = @rkppl begin
            mu_alpha ~ Normal(0, 10)
            sigma_alpha ~ Exponential(1)
            s ~ Exponential(1)
            c[levels(g)] .~ Normal.(mu_alpha, sigma_alpha)
            mu = c[g]
            y .~ Normal.(mu, s)
        end
        overlay = quote
            z[levels(g)] .~ Normal.(0, 1)
            c = mu_alpha .+ sigma_alpha .* z
        end
        hand = @rkppl begin
            mu_alpha ~ Normal(0, 10)
            sigma_alpha ~ Exponential(1)
            s ~ Exponential(1)
            c = mu_alpha .+ sigma_alpha .* z
            mu = c[g]
            y .~ Normal.(mu, s)
            z[levels(g)] .~ Normal.(0, 1)
        end
        _merge_respelled_checks(centred, overlay, hand, :c, d -> d.sigma_alpha, g, y)
    end
    # A positive effect, which adaptive centring does not take: its support
    # transform is not the identity.
    @testset "a positive effect per level" begin
        centred = @rkppl begin
            m ~ Normal(0, 1)
            t ~ Exponential(1)
            s ~ Exponential(1)
            r[levels(g)] .~ LogNormal.(m, t)
            mu = r[g]
            y .~ Normal.(mu, s)
        end
        overlay = quote
            z[levels(g)] .~ Normal.(0, 1)
            r = exp.(m .+ t .* z)
        end
        hand = @rkppl begin
            m ~ Normal(0, 1)
            t ~ Exponential(1)
            s ~ Exponential(1)
            r = exp.(m .+ t .* z)
            mu = r[g]
            y .~ Normal.(mu, s)
            z[levels(g)] .~ Normal.(0, 1)
        end
        _merge_respelled_checks(centred, overlay, hand, :r, d -> d.t, g, y)
    end
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
    bf = (fixed(; x = cols[:x]) | (; y = cols[:y]))
    @test bf.columns[:s] == fill(2.0, n)
    bh = (hand(; x = cols[:x], s = fill(2.0, n)) | (; y = cols[:y]))
    # A pin is a whole model-level array; the handwritten input is a row column.
    # Both spellings retain the same mathematics and free coordinates.
    u = [0.2, -0.1]
    @test build_kernel(bf).layout.total == build_kernel(bh).layout.total
    @test _query(build_kernel(bf).spec, bf, :posterior, u) ≈
        _query(build_kernel(bh).spec, bh, :posterior, u)
    # Explicit kwargs win over fixed (SB easily-rebound data).
    rebound = (fixed(; x = cols[:x], s = ones(n)) | (; y = cols[:y]))
    @test rebound.columns[:s] == ones(n)
    # Fix-wins over a splice naming the same LHS in one call.
    both = Base.merge(merge_base, :(s ~ merge_latent_normal(2.0)),
        (; s = fill(2.0, n)))
    @test both.fixed[:s] == fill(2.0, n)
    bb = (both(; x = cols[:x]) | (; y = cols[:y]))
    @test _query(build_kernel(bb).spec, bb, :posterior, u) ≈
        _query(build_kernel(bh).spec, bh, :posterior, u)
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
    # refused: merge override is not a statement (P9, 18h1h54: merge replaces a statement)
    @test_throws SurfaceLoweringError Base.merge(merge_base, :(x + 1))
    # refused: bare symbol is not a statement (P9, 18h1h54)
    @test_throws SurfaceLoweringError Base.merge(merge_base, :b)
    # A partial element write names no complete declaration.
    @test_throws SurfaceLoweringError Base.merge(merge_base, :(c[1] = 2))
    # refused: a ranged observation override double-observes y (10gzbm9 merge-partial)
    @test_throws SurfaceLoweringError Base.merge(merge_base,
        Expr(:call, :.~, :(y[1:3]), :(Normal.(mu, s))))
    # Julia blocks are transparent during statement splicing.
    @test _plans_equal(_merge_lower(Base.merge(merge_base,
        quote
            begin
                a ~ Normal(0, 10)
            end
        end)), _merge_lower(Base.merge(merge_base, :(a ~ Normal(0, 10)))))
    # A broken base (duplicate LHS) fails closed, naming the name.
    dup = RKPPLModel(quote
        a ~ Normal(0, 1)
        a ~ Normal(0, 2)
        y .~ Normal.(a, 1.0)
    end, @__MODULE__)
    # refused: single assignment: base binds a twice
    @test_throws SurfaceLoweringError Base.merge(dup, :(a ~ Normal(0, 1)))
    # A bare stem replaces the complete indexed declaration.
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
    # capability: replace an indexed declaration with a scalar statement; check uses at bind (P9 18h1h54; todo `1qlbn5b`).
    @test err === nothing

    # refused: a fix must name a statement to remove (merge docstring).
    # refused: pin names no model statement (unknown name)
    @test_throws SurfaceLoweringError Base.merge(merge_base, (; nosuch = ones(6)))
    fixed_array = Base.merge(lev, (; c = ones(6)))
    @test fixed_array.fixed[:c] == ones(6)
    @test !occursin("c[levels", sprint(Base.show_unquoted, fixed_array.ast))
    # A plate-cell local has its own scope; appending a top-level definition
    # of the same name preserves the cell's authored value.
    plated = RKPPLModel(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(4),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:block, LineNumberNode(5),
                        :(t = a .+ b .* x[i]),
                        :(y[i] ~ Normal.(t, s)))))), @__MODULE__)
    shadowed = Base.merge(plated, :(t = a .+ b .* x))
    cols = Dict(:y => [-0.1, 0.8, 0.2], :x => [0.3, 0.5, -0.2])
    bound = bind_data(lower_rkppl(shadowed.ast, cols; conditioned = keys(cols)), cols)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a = 0.2, b = -0.3, s = 1.1))
    oracle = v -> begin
        q = constrain(built.layout, v)
        logpdf(Normal(), q.a) + logpdf(Normal(0, 2), q.b) +
            logpdf(Exponential(), q.s) + log(q.s) +
            sum(logpdf.(Normal.(q.a .+ q.b .* cols[:x], q.s), cols[:y]))
    end
    _check_model_math(built, bound, u, oracle)
end

@testset "merge joint session over shared defs" begin
    m1 = @rkppl begin
        b ~ Normal(0, 1)
        a ~ Normal(0, 1)
        s ~ merge_latent_normal(1.0)
        mu = a .+ b .* x
        y ~ merge_normal_stream(mu, s)
    end
    m2 = @rkppl begin
        b ~ Normal(0, 1)
        a ~ Normal(0, 3)
        t ~ merge_latent_normal(2.0)
        nu = a .+ b .* x
        y ~ merge_normal_stream(nu, t)
    end
    p1 = _merge_lower(m1)
    p2 = _merge_lower(m2)
    @test any(p -> _test_scope_name(p1, p.name) === :s_r, p1.parameters)
    @test !any(p -> _test_scope_name(p1, p.name) === :t_r, p1.parameters)
    @test any(p -> _test_scope_name(p2, p.name) === :t_r, p2.parameters)
    @test !any(p -> _test_scope_name(p2, p.name) === :s_r, p2.parameters)
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
    bm = (merged(; x = cols[:x]) | (; y = cols[:y]))
    bh = (hand(; x = cols[:x]) | (; y = cols[:y]))
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
    bf = (fixed(; x = cols[:x]) | (; y = cols[:y]))
    bhf = (handf(; x = cols[:x], s = fill(2.0, n)) | (; y = cols[:y]))
    @test build_kernel(bf).layout.total == build_kernel(bhf).layout.total
    uf = [0.5, -0.25]
    @test _query(build_kernel(bf).spec, bf, :posterior, uf) ==
          _query(build_kernel(bhf).spec, bhf, :posterior, uf)
end

@testset "merge pins a number" begin
    # A pinned number reads as the definition it replaces (decisions
    # 18h1h54, 0dejlw1): `s` drops its submodel prior and scales the stream.
    X = [0.5, -1.0, 1.5, 0.0]
    Y = [0.3, -0.8, 1.9, 0.2]
    bound = Base.merge(merge_base, (; s = 2.0))(; x = X) | (; y = Y)
    built = build_kernel(bound)
    u = [0.2, -0.4]
    values = ReactiveKernelsPPL.constrain(built.layout, u)
    a, b = values.a, values.b
    @test Base.invokelatest(prepare_query(built, bound, :likelihood), u) ≈
        sum(logpdf.(Normal.(a .+ b .* X, 2.0), Y))
end
