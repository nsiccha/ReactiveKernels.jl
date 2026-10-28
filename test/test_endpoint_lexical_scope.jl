module EndpointLexicalScopeTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# A constructed endpoint inside a generator term, closure, loop or `let` body
# is evaluated where it is written, each time Julia evaluates that position,
# and reads the names the scope binds. It used to be spliced above the scope:
# `m[s]` became a top-level recipe, so the first evaluation raised
# `UndefVarError: s not defined`, or, with a global of that name, silently
# read the global (snag `rk-endpoint-call-d125dbaf`). This module's `s`
# global must never be read in place of a generator variable.
s = 1

nlp(x, mu, sigma) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2pi)

@kernel generator_sum(x, m) = begin
    total = sum(normal(m[s], 1.0).logpdf(x[1]) for s in 1:2; init = 0.0)
    return total
end

@kernel bare_variable(x, m) = begin
    total = sum(normal(m[s], s).logpdf(x[1]) for s in eachindex(m); init = 0.0)
    return total
end

@kernel filtered(x, m) = begin
    total = sum(normal(m[s], 1.0).logpdf(x[1]) for s in eachindex(m) if m[s] > 0.5;
                init = 0.0)
    return total
end

@kernel nested_generator(x, m) = begin
    total = sum(normal(m[s], 1.0).logpdf(x[i])
                for i in eachindex(x) for s in eachindex(m); init = 0.0)
    return total
end

@kernel shadowing(x, m) = begin
    # The generator's `x` is not the caller port `x`.
    total = sum(normal(0.0, 1.0).logpdf(x) for x in m; init = 0.0)
    return total
end

@kernel zero_trip(x, m) = begin
    # An invalid scale is never evaluated when the generator is empty.
    total = sum(normal(m[s], -1.0).logpdf(x[1]) for s in 1:0; init = 0.0)
    return total
end

@kernel comprehension(x, m) = begin
    terms = [normal(m[s], 1.0).logpdf(x[1]) for s in eachindex(m)]
    total = sum(terms)
    return total
end

@kernel closure(x, m) = begin
    total = sum(map(s -> normal(m[s], 1.0).logpdf(x[1]), eachindex(m)))
    return total
end

@kernel do_block(x, m) = begin
    total = sum(map(eachindex(m)) do s
        mu = m[s]
        normal(mu, 1.0).logpdf(x[1])
    end)
    return total
end

@kernel loop(x, m) = begin
    total = begin
        acc = 0.0
        for s in eachindex(m)
            mu = m[s]
            acc += normal(mu, 1.0).logpdf(x[1])
        end
        acc
    end
    return total
end

@kernel let_body(x, m) = begin
    total = let mu = m[2]
        normal(mu, 1.0).logpdf(x[1])
    end
    return total
end

@kernel local_function(x, m) = begin
    total = begin
        f(s) = normal(m[s], 1.0).logpdf(x[1])
        f(1) + f(2)
    end
    return total
end

@kernel plate_cell(x, m) = begin
    cells = plate(x) do xi
        sum(normal(m[s], 1.0).logpdf(xi) for s in eachindex(m); init = 0.0)
    end
    total = sum(cells)
    return total
end

@kernel scan_step(x, m) = begin
    out = scan(x; init = 0.0) do carry, xi
        e = sum(normal(m[s], 1.0).logpdf(xi) for s in eachindex(m); init = 0.0)
        (carry + e, e)
    end
    total = sum(out)
    return total
end

const X = [0.3, 1.2, -0.4]
const M = [0.0, 1.0]
first_point(m) = sum(nlp(X[1], mu, 1.0) for mu in m)
every_point(m) = sum(nlp(xi, mu, 1.0) for xi in X, mu in m)

@testset "endpoint calls read the names of their scope" begin
    x1 = X[1:1]
    cases = (
        (generator_sum, x1, first_point(M)),
        (bare_variable, x1, nlp(X[1], M[1], 1.0) + nlp(X[1], M[2], 2.0)),
        (filtered, x1, nlp(X[1], M[2], 1.0)),
        (nested_generator, X, every_point(M)),
        (shadowing, X, sum(nlp(mu, 0.0, 1.0) for mu in M)),
        (zero_trip, x1, 0.0),
        (comprehension, x1, first_point(M)),
        (closure, x1, first_point(M)),
        (do_block, x1, first_point(M)),
        (loop, x1, first_point(M)),
        (let_body, x1, nlp(X[1], M[2], 1.0)),
        (local_function, x1, first_point(M)),
        (plate_cell, X, every_point(M)),
        (scan_step, X, every_point(M)),
    )
    for (spec, x, expected) in cases
        saved = (copy(x), copy(M))
        @test prepare(spec)(x, M) ≈ expected rtol = 1e-14
        @test (x, M) == saved
    end
    # Nothing leaves the generator: its endpoint is one in-place `let`.
    g = kernel_graph(generator_sum)
    @test length(g.recipes) == 1
    @test occursin("m[s]", sprint(show, MIME"text/plain"(), g))
end

@testset "endpoint calls in scopes keep ordinary native reverse" begin
    reverse = AutoEnzyme(; mode = Enzyme.Reverse)
    # d/dm[s] of a sum of unit normal log densities at x: (x - m[s]).
    cases = (
        (generator_sum, X[1:1], [X[1] - M[1], X[1] - M[2]]),
        (filtered, X[1:1], [0.0, X[1] - M[2]]),
        (loop, X[1:1], [X[1] - M[1], X[1] - M[2]]),
        (local_function, X[1:1], [X[1] - M[1], X[1] - M[2]]),
        (plate_cell, X, [sum(X .- M[1]), sum(X .- M[2])]),
        (scan_step, X, [sum(X .- M[1]), sum(X .- M[2])]),
    )
    for (spec, x, expected) in cases, bound in (false, true)
        k = bound ? prepare(spec; bound = (; x)) : prepare(spec)
        args = bound ? (M,) : (x, M)
        ad = prepare_ad(k, reverse, args...; active = :m)
        @test ad_gradient(ad, args...) ≈ expected rtol = 1e-12
    end
end
end
