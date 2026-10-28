using Test, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme

_cat_q(built, bound, query, u) =
    Base.invokelatest(prepare_query(built, bound, query), u)

@testset "computed categorical probabilities in plate cells" begin
    # Both public response shapes represent the same categorical law: one
    # vector per group, or one scalar observation selecting its group.
    document = quote
        eachcol(theta[1:2, 1:M]) .~ Dirichlet(ones(2))
        eachcol(phi[1:V, 1:2]) .~ Dirichlet(ones(V))
        @plate for m in eachindex(words)
            words[m] .~ Categorical(phi * theta[:, m])
        end
    end
    scalar = quote
        eachcol(theta[1:2, 1:M]) .~ Dirichlet(ones(2))
        eachcol(phi[1:V, 1:2]) .~ Dirichlet(ones(V))
        @plate for i in eachindex(w)
            w[i] ~ Categorical(phi * theta[:, doc[i]])
        end
    end
    named_cell = quote
        eachcol(theta[1:2, 1:M]) .~ Dirichlet(ones(2))
        eachcol(phi[1:V, 1:2]) .~ Dirichlet(ones(V))
        @plate for i in eachindex(w)
            probabilities = phi * theta[:, doc[i]]
            w[i] ~ Categorical(probabilities)
        end
    end
    document_data = (; M=2, V=3, words=[[1, 2], [3]])
    scalar_data = (; M=2, V=3, doc=[1, 1, 2], w=[1, 2, 3])
    results = Any[]
    for (ast, data) in ((document, document_data),
            (scalar, scalar_data), (named_cell, scalar_data))
        saved = deepcopy(data)
        bound = bind_data(lower_rkppl(ast, data;
            conditioned=keys(data)), data)
        built = build_kernel(bound)
        u = [0.2, -0.3, 0.1, -0.2, 0.3, -0.4]
        q = constrain(built.layout, u)
        oracle = sum(logpdf(Categorical(q.phi * q.theta[:, m]), y)
            for (m, ys) in enumerate(document_data.words) for y in ys)
        likelihood = _cat_q(built, bound, :likelihood, u)
        @test likelihood ≈ oracle atol=1e-12
        sampler = prepare_sampler(built, bound, u;
            backend=DifferentiationInterface.AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        direct = v -> _cat_q(built, bound, :sampler, v)
        finite_difference = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += 1e-5
            minus[j] -= 1e-5
            (direct(plus) - direct(minus)) / 2e-5
        end
        @test value ≈ direct(u) atol=1e-12
        @test gradient ≈ finite_difference atol=2e-6 rtol=2e-5
        @test data == saved
        push!(results, (value, likelihood, gradient))
    end
    @test results[1][1] ≈ results[2][1] atol=1e-12
    @test results[1][2] ≈ results[2][2] atol=1e-12
    @test results[1][3] ≈ results[2][3] atol=1e-10 rtol=1e-10
    @test results[2][1] ≈ results[3][1] atol=1e-12
    @test results[2][2] ≈ results[3][2] atol=1e-12
    @test results[2][3] ≈ results[3][3] atol=1e-10 rtol=1e-10

    shared = quote
        p ~ Dirichlet(ones(3))
        @plate for i in eachindex(y)
            y[i] ~ Categorical(p)
        end
    end
    shared_data = (; y=[1, 2, 3])
    shared_bound = bind_data(lower_rkppl(shared, shared_data;
        conditioned=keys(shared_data)), shared_data)
    shared_built = build_kernel(shared_bound)
    shared_u = [0.1, -0.2]
    shared_p = constrain(shared_built.layout, shared_u).p
    @test _cat_q(shared_built, shared_bound, :likelihood, shared_u) ≈
        sum(logpdf(Categorical(shared_p), y) for y in shared_data.y)

    invalid = quote
        p ~ Dirichlet(ones(2))
        @plate for i in eachindex(y)
            y[i] ~ Categorical(p .* 0.5)
        end
    end
    invalid_data = (; y=[1, 2])
    invalid_bound = bind_data(lower_rkppl(invalid, invalid_data;
        conditioned=keys(invalid_data)), invalid_data)
    invalid_built = build_kernel(invalid_bound)
    # refused: probabilities summing to 0.5 define no categorical law, so
    # scoring them would return a meaningless density (dev §1: no silent
    # defaults for invalid inputs).
    @test_throws DomainError _cat_q(invalid_built, invalid_bound,
        :likelihood, [0.1])

    # An observed category outside the vector has zero probability, with the
    # probabilities' own floating-point type.
    @test ReactiveKernelsPPL._ppl_categorical_logdensity(3, [0.5, 0.5]) ===
        -Inf
    @test ReactiveKernelsPPL._ppl_categorical_logdensity(0, Float32[0.5, 0.5]) ===
        -Inf32

    outside_data = (; M=2, V=3, doc=[3], w=[1])
    outside_bound = bind_data(lower_rkppl(scalar, outside_data;
        conditioned=keys(outside_data)), outside_data)
    outside_built = build_kernel(outside_bound)
    # refused: doc[1] = 3 names a third theta column that M = 2 does not
    # allocate; Julia indexing semantics must surface the invalid read.
    @test_throws BoundsError _cat_q(outside_built, outside_bound,
        :likelihood, zeros(length(coordinate_names(outside_built.layout))))
end

# Counts how often a model computes a categorical probability vector.
const _CATEGORICAL_EVALUATIONS = Ref(0)
_counted_mix(P, t) = (_CATEGORICAL_EVALUATIONS[] += 1; P * t)

@testset "computed categorical vectors: one per key, one check per vector" begin
    prior = quote
        eachcol(theta[1:2, 1:M]) .~ Dirichlet(ones(2))
        eachcol(phi[1:V, 1:2]) .~ Dirichlet(ones(V))
    end
    model(cell) = Expr(:block, prior.args..., cell.args...)
    # A word reads its document only through doc[i], so every word of one
    # document scores the same vector: it is computed once per document.
    keyed = model(quote
        @plate for i in eachindex(w)
            w[i] ~ Categorical(_counted_mix(phi, theta[:, doc[i]]))
        end
    end)
    # Reading the index through two columns keeps one vector per word.
    two_keys = model(quote
        @plate for i in eachindex(w)
            w[i] ~ Categorical(_counted_mix(phi,
                0.5 .* theta[:, doc[i]] .+ 0.5 .* theta[:, alt[i]]))
        end
    end)
    data = (; M=3, V=3, doc=[2, 1, 2, 3, 2, 1, 3], alt=[1, 1, 3, 2, 2, 3, 1],
        w=[1, 3, 2, 2, 3, 1, 1])
    u = [0.2, -0.3, 0.1, -0.2, 0.3, -0.4, 0.15]
    for (ast, location, evaluations) in ((keyed, (q, i) -> q.theta[:, data.doc[i]], 3),
            (two_keys, (q, i) -> 0.5 .* q.theta[:, data.doc[i]] .+
                0.5 .* q.theta[:, data.alt[i]], length(data.w)))
        saved = deepcopy(data)
        bound = bind_data(lower_rkppl(ast, data; conditioned=(:w,),
            mod=@__MODULE__), data)
        built = build_kernel(bound)
        q = constrain(built.layout, u)
        oracle = [logpdf(Categorical(q.phi * location(q, i)), y)
            for (i, y) in enumerate(data.w)]
        likelihood = prepare_query(built, bound, :likelihood)
        Base.invokelatest(likelihood, u)
        _CATEGORICAL_EVALUATIONS[] = 0
        @test Base.invokelatest(likelihood, u) ≈ sum(oracle) atol=1e-12
        @test _CATEGORICAL_EVALUATIONS[] == evaluations
        @test _cat_q(built, bound, :pointwise, u).w ≈ oracle atol=1e-12
        sampler = prepare_sampler(built, bound, u;
            backend=DifferentiationInterface.AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        direct = v -> _cat_q(built, bound, :sampler, v)
        finite_difference = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += 1e-5
            minus[j] -= 1e-5
            (direct(plus) - direct(minus)) / 2e-5
        end
        @test value ≈ direct(u) atol=1e-12
        @test gradient ≈ finite_difference atol=2e-6 rtol=2e-5
        @test data == saved
    end

    # refused: probabilities summing to 0.5 define no categorical law; the
    # check runs once per computed vector, and still runs (dev §1).
    halved = model(quote
        @plate for i in eachindex(w)
            w[i] ~ Categorical(0.5 .* (phi * theta[:, doc[i]]))
        end
    end)
    halved_bound = bind_data(lower_rkppl(halved, data; conditioned=(:w,)), data)
    @test_throws DomainError _cat_q(build_kernel(halved_bound), halved_bound,
        :likelihood, u)

    # A document's words share its vector, checked once for all of them.
    document = model(quote
        @plate for m in eachindex(words)
            words[m] .~ Categorical(phi * theta[:, m])
        end
    end)
    halved_document = model(quote
        @plate for m in eachindex(words)
            words[m] .~ Categorical(0.5 .* (phi * theta[:, m]))
        end
    end)
    words = [[1, 3], [2, 2, 3], Int[]]
    document_data = (; M=3, V=3, words)
    document_bound = bind_data(lower_rkppl(document, document_data;
        conditioned=(:words,)), document_data)
    document_built = build_kernel(document_bound)
    q = constrain(document_built.layout, u)
    @test _cat_q(document_built, document_bound, :likelihood, u) ≈
        sum(logpdf(Categorical(q.phi * q.theta[:, m]), y)
            for (m, ys) in enumerate(words) for y in ys) atol=1e-12
    halved_document_bound = bind_data(lower_rkppl(halved_document,
        document_data; conditioned=(:words,)), document_data)
    @test_throws DomainError _cat_q(build_kernel(halved_document_bound),
        halved_document_bound, :likelihood, u)

    # No words: no vectors, zero likelihood.
    empty_data = (; M=3, V=3, doc=Int[], w=Int[])
    empty_bound = bind_data(lower_rkppl(keyed, empty_data; conditioned=(:w,),
        mod=@__MODULE__), empty_data)
    _CATEGORICAL_EVALUATIONS[] = 0
    @test _cat_q(build_kernel(empty_bound), empty_bound, :likelihood, u) == 0
    @test _CATEGORICAL_EVALUATIONS[] == 0
end

# A data matrix that counts the matrix-vector products it takes part in.
struct _CountedColumns{M<:AbstractMatrix{Float64}} <: AbstractMatrix{Float64}
    parent::M
end
Base.size(A::_CountedColumns) = size(A.parent)
Base.getindex(A::_CountedColumns, i::Int, j::Int) = A.parent[i, j]
const _CATEGORICAL_PRODUCTS = Ref(0)
Base.:*(A::_CountedColumns, x::AbstractVector) =
    (_CATEGORICAL_PRODUCTS[] += 1; A.parent * x)

@testset "categorical matrix-column products score observed entries" begin
    # `P * theta[:, g]` scores each category as `log(dot(P[y, :], theta[:, g]))`
    # without forming the vector; its check reads the column sums of `P`.
    keyed = quote
        eachcol(theta[1:2, 1:M]) .~ Dirichlet(ones(2))
        @plate for i in eachindex(w)
            w[i] ~ Categorical(P * theta[:, doc[i]])
        end
    end
    document = quote
        eachcol(theta[1:2, 1:M]) .~ Dirichlet(ones(2))
        @plate for m in eachindex(words)
            words[m] .~ Categorical(P * theta[:, m])
        end
    end
    P = [0.5 0.1; 0.3 0.2; 0.2 0.7]
    doc = [2, 1, 2, 3, 1]
    w = [1, 3, 2, 2, 3]
    words = [w[doc .== m] for m in 1:3]
    u = [0.2, -0.3, 0.4]
    for (ast, data) in ((keyed, (; M=3, P=_CountedColumns(P), doc, w)),
            (document, (; M=3, P=_CountedColumns(P), words)))
        saved = deepcopy(data)
        conditioned = haskey(data, :w) ? (:w,) : (:words,)
        bound = bind_data(lower_rkppl(ast, data; conditioned), data)
        built = build_kernel(bound)
        q = constrain(built.layout, u)
        oracle = [logpdf(Categorical(P * q.theta[:, doc[i]]), y)
            for (i, y) in enumerate(w)]
        _CATEGORICAL_PRODUCTS[] = 0
        @test _cat_q(built, bound, :likelihood, u) ≈ sum(oracle) atol=1e-12
        # Valid nonnegative columns: no vector is formed.
        @test _CATEGORICAL_PRODUCTS[] == 0
        haskey(data, :w) &&
            @test _cat_q(built, bound, :pointwise, u).w ≈ oracle atol=1e-12
        sampler = prepare_sampler(built, bound, u;
            backend=DifferentiationInterface.AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        direct = v -> _cat_q(built, bound, :sampler, v)
        finite_difference = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += 1e-5
            minus[j] -= 1e-5
            (direct(plus) - direct(minus)) / 2e-5
        end
        @test value ≈ direct(u) atol=1e-12
        @test gradient ≈ finite_difference atol=2e-6 rtol=2e-5
        @test data == saved
    end

    # A category outside the vector has zero probability.
    outside = (; M=3, P=_CountedColumns(P), doc=[1], w=[4])
    outside_bound = bind_data(lower_rkppl(keyed, outside; conditioned=(:w,)), outside)
    @test _cat_q(build_kernel(outside_bound), outside_bound, :likelihood, u) === -Inf

    # refused: columns summing to one half define no categorical law (dev §1);
    # the check still runs without forming the vector, and raises with it.
    halved = (; M=3, P=_CountedColumns(0.5 .* P), doc, w)
    halved_bound = bind_data(lower_rkppl(keyed, halved; conditioned=(:w,)), halved)
    @test_throws DomainError _cat_q(build_kernel(halved_bound), halved_bound,
        :likelihood, u)

    # A negative entry leaves the sign to the vector itself: it is formed
    # once per checked column, and a valid one scores as before.
    mixed = [0.6 0.0; 0.5 0.1; -0.1 0.9]
    weights = [1.0 0.5; 0.0 0.5]
    fixed = quote
        s ~ Normal(0, 1)
        @plate for i in eachindex(w)
            w[i] ~ Categorical(P * W[:, doc[i]])
        end
    end
    valid = (; P=_CountedColumns(mixed), W=weights, doc=[2, 2, 2], w=[1, 2, 3])
    valid_bound = bind_data(lower_rkppl(fixed, valid; conditioned=(:w,)), valid)
    _CATEGORICAL_PRODUCTS[] = 0
    @test _cat_q(build_kernel(valid_bound), valid_bound, :likelihood, [0.1]) ≈
        sum(log((mixed * weights[:, 2])[y]) for y in valid.w) atol=1e-12
    @test _CATEGORICAL_PRODUCTS[] == 1
    # refused: column 1 gives the third category probability -0.1 (dev §1).
    negative = (; P=_CountedColumns(mixed), W=weights, doc=[2, 1], w=[1, 2])
    negative_bound = bind_data(lower_rkppl(fixed, negative; conditioned=(:w,)), negative)
    @test_throws DomainError _cat_q(build_kernel(negative_bound), negative_bound,
        :likelihood, [0.1])
    # refused: `P * W[:, j]` needs one weight per column of `P`, as in Julia.
    short = (; P=_CountedColumns(mixed), W=ones(1, 2), doc=[1], w=[1])
    short_bound = bind_data(lower_rkppl(fixed, short; conditioned=(:w,)), short)
    @test_throws DimensionMismatch _cat_q(build_kernel(short_bound), short_bound,
        :likelihood, [0.1])
end
