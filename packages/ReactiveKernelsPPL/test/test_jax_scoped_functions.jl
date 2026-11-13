using ReactiveKernels, ReactiveKernelsPPL, Test
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

@rkppl jax_scope_leaf(loc, scale) = begin
    "Local deviation."
    z ~ Normal(loc, scale)
    return z + 1
end

@rkppl jax_scope_parent(loc, scale) = begin
    left ~ jax_scope_leaf(loc, scale)
    right ~ jax_scope_leaf(loc, scale)
    return left + right
end

@rkppl jax_scope_unsafe(n, loc) = begin
    z[1:n] .~ Normal.(loc, 1.0)
    return z[1]
end

function jax_scope_check(bound, raw; entries = ((:log_density, :sampler),), source = nothing)
    built = build_kernel(bound)
    src = source === nothing ? jax_source(built, bound) : source
    names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
    data = Dict(replace(String(k), "." => "_") => v for (k, v) in pairs(raw))
    args = Tuple(data[name] for name in names)
    u = [0.1sin(i) for i in 1:built.layout.total]
    if rk_jax_available()
        for (entry, workflow) in entries
            native = prepare_query(built, bound, workflow)
            got = rk_jax_evaluate(src, entry, (u, args...); grad = true, prepare = :arguments)
            @test got.value ≈ native(u) atol = 1e-10
            fd = [(native(u + 1e-5 .* (eachindex(u) .== i)) -
                   native(u - 1e-5 .* (eachindex(u) .== i))) / 2e-5 for i in eachindex(u)]
            @test got.grad ≈ fd rtol = 2e-6 atol = 1e-7
        end
    else
        @test_skip false
    end
    src, built
end

@testset "nested submodels, pins and observations" begin
    model = @rkppl begin
        a ~ jax_scope_parent(loc, scale)
        b ~ jax_scope_parent(loc, scale)
        y .~ Normal.(a + b, 1.0)
    end
    raw = (; loc = 0.2, scale = 1.3, y = [0.1, -0.2, 0.4])
    entries = ((:log_density, :sampler), (:log_prior, :prior), (:log_likelihood, :likelihood))
    src, _ = jax_scope_check(model(; raw.loc, raw.scale) | (; raw.y), raw; entries)
    @test count(r"(?m)^def jax_scope_parent\w*\(", src.text) == 1
    @test count(r"(?m)^def jax_scope_leaf\w*\(", src.text) == 1
    fixed = (; var"a.left.z" = 0.3)
    observed = model(; raw.loc, raw.scale) | (; raw.y, fixed...)
    jax_scope_check(observed, (; raw..., fixed...); entries)
    pinned = merge(model, fixed)(; raw.loc, raw.scale) | (; raw.y)
    jax_scope_check(pinned, (; raw..., fixed...); entries)
end

@testset "literal and expression submodel arguments" begin
    literal = @rkppl begin
        a ~ jax_scope_leaf(0.0, 1.0)
        b ~ jax_scope_leaf(2.0, 3.0)
        y .~ Normal.(a + b, 1.0)
    end
    expression = @rkppl begin
        a ~ jax_scope_leaf(loc + 1.0, scale)
        b ~ jax_scope_leaf(loc + 2.0, scale)
        y .~ Normal.(a + b, 1.0)
    end
    raw = (; loc = 0.4, scale = 1.3, y = [0.1, 0.3])
    entries = ((:log_density, :sampler), (:log_prior, :prior), (:log_likelihood, :likelihood))
    for bound in (literal() | (; raw.y), expression(; raw.loc, raw.scale) | (; raw.y))
        src, _ = jax_scope_check(bound, raw; entries)
        @test count(r"(?m)^def jax_scope_leaf\w*\(", src.text) == 1
    end
end

@testset "unused submodel value preserves access safety" begin
    model = @rkppl begin
        a ~ jax_scope_unsafe(n, loc)
        y .~ Normal.(a, 1.0)
    end
    raw = (; n = 0, loc = 0.3, y = [0.1])
    bound = model(; raw.n, raw.loc) | (; raw.y)
    jax_scope_check(bound, raw; entries = ((:log_prior, :prior),))
end

@testset "submodel calls in ragged plates" begin
    model = @rkppl begin
        scale ~ Exponential(rate)
        @plate for i in eachindex(x)
            a[i] ~ jax_scope_leaf(sum(x[i]), scale)
            y[i] .~ Normal.(a[i], 1.0)
        end
    end
    source = nothing
    programs = Expr[]
    for x in ([[0.2, 0.4], Float64[], [-0.1]],
              [[0.2], [0.4, 0.1, -0.2], Float64[], [0.8, -0.2]])
        raw = (; x, rate = 1.2, y = [zero.(row) for row in x])
        bound = model(; raw.x, raw.rate) | (; raw.y)
        src, built = jax_scope_check(bound, raw; source)
        @test occursin(r"(?m)^def jax_scope_leaf\(", src.text)
        body = match(r"(?ms)^def jax_scope_leaf\(.*?(?=^def |\z)", src.text)
        @test body !== nothing && occursin(r"value\w* = z\w*(?:\[[^\n]+\])? \+ 1", body.match)
        source = src
        push!(programs, built.program)
    end
    @test ReactiveKernelsPPL._same_program(programs...)
end
