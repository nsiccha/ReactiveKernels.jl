module EmittedNameTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test

# The generated program reads with the model's own names (rkppl-use §16):
# plate cells name their arguments after the values they iterate, a value
# the lowering computes for a distribution argument is named after its
# owner and role, a declared array's prior is named after the array, sums
# carry no `0.0` seed, `Float64` values need no `1.0 *` port and the
# display spells the model module's functions as written. Every density,
# gradient and coordinate stays the same.

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

module Fns
spread(loc, a, p) = sqrt.(a^2 .+ (loc .* p) .^ 2)
shiftby(v, by) = v .+ by
end
using .Fns: spread, shiftby

const x = [0.3, -0.5, 1.2, 0.8]
const y = [0.1, -0.4, 1.1, 0.6]
const g = [1, 2, 1, 2]

_text(bound) = string(readable_code(kernel_expr(bound, assign_layout(bound))))
_bound(ast, data; mod = Fns) =
    bind_data(lower_rkppl(ast, data; conditioned = (:y,), mod), data)

const MODEL = quote
    a ~ Normal(0, 1)
    sa ~ Exponential(4.0)
    sp ~ Exponential(4.0)
    s ~ Exponential(1.0)
    b ~ Normal(0, 2 * s)
    tau[1:2] .~ truncated.(Normal.(0, 1), 0, Inf)
    z[1:2, levels(g)] .~ Normal.(0, 1)
    X = hcat(ones(length(x)), x)
    c[axes(X, 2)] .~ Normal.([1.0, 0.0], [0.8, 1.0])
    r = z .* tau
    loc = a .+ b .* x .+ X * c .+ r[1, g]
    y .~ Normal.(loc, spread(loc, sa, sp))
end

function _oracle(nt)
    loc = nt.a .+ nt.b .* x .+ hcat(ones(length(x)), x) * nt.c .+
        (nt.z .* nt.tau)[1, g]
    sd = Fns.spread(loc, nt.sa, nt.sp)
    lik = sum(logpdf.(Normal.(loc, sd), y))
    prior = logpdf(Normal(0, 1), nt.a) + logpdf(Exponential(4.0), nt.sa) +
        logpdf(Exponential(4.0), nt.sp) + logpdf(Exponential(1.0), nt.s) +
        logpdf(Normal(0, 2 * nt.s), nt.b) +
        sum(logpdf.(truncated(Normal(0, 1), 0, Inf), nt.tau)) +
        sum(logpdf.(Normal(0, 1), nt.z)) +
        sum(logpdf.(Normal.([1.0, 0.0], [0.8, 1.0]), nt.c))
    return lik, prior
end

@testset "generated programs read with the model's own names" begin
    data = (; x, y, g)
    bound = _bound(MODEL, data)
    text = _text(bound)

    @testset "plate cells name their arguments after their values" begin
        # `loc` is a retained value (`spread` reads it too); the location
        # `y_eta = loc` aliases it, and the cell argument takes its name.
        @test occursin("plate(y, y_eta, y_scale) do y, loc, y_scale", text)
        @test occursin("(normal(loc, y_scale)).logpdf(y)", text)
        @test occursin("plate(y, loc, y_scale) do y, loc, y_scale",
            string(readable_code(build_kernel(bound).spec)))
        @test !occursin(r"_ppl_c\d", text)
    end

    @testset "a computed distribution argument is named after its role" begin
        @test occursin("y_scale = spread(loc, sa, sp)", text)
        @test occursin("b_scale = 2s", text)
        @test !occursin("_rkppl_", text)
    end

    @testset "a prior cell reads a generated argument by its role" begin
        @test occursin("plate(c, _ppl_parg_c_1, _ppl_parg_c_2) do c, location, scale", text)
        @test occursin("(normal(location, scale)).logpdf(c)", text)
    end

    @testset "a two-axis array's prior is named after the array" begin
        @test occursin("var\"_ppl_prior_z\"", text) || occursin("_ppl_prior_z::", text)
        @test occursin("plate(_ppl_arrflat_z) do z", text)
        @test !occursin("_ppl_prior__ppl_arrflat", text)
    end

    @testset "sums carry no zero seed; a zero floor is exp alone" begin
        @test occursin("likelihood::Float64 = _ppl_lik_y_resp\n", text)
        @test occursin("prior::Float64 = ((((((", text) || occursin("prior::Float64 = (", text)
        @test !occursin("0.0 +", text)
        @test !occursin("0.0 .+", text)
        @test occursin("tau::AbstractVector{Float64} = exp.(_ppl_floor_tau)", text)
    end

    @testset "Float64 prior arguments need no port promotion" begin
        @test occursin("(exponential(1.0)).logpdf(s)", text)
        @test !occursin("1.0location", text) && !occursin("1.0scale", text)
        # A computed value's type is not proven, so it keeps its promotion.
        @test occursin("(normal(0.0, 1.0b_scale)).logpdf(b)", text)
    end

    @testset "the display spells the model module's functions as written" begin
        code = readable_code(kernel_expr(bound, assign_layout(bound)))
        @test Fns in code.modules
        @test startswith(string(code), "# authored sources evaluated in: ")
        @test !occursin("Fns.spread", text)
        # `kernel_expr` itself keeps the exact global reference.
        @test occursin("Fns.spread", sprint(print, kernel_expr(bound, assign_layout(bound))))
    end

    @testset "values and native gradients are the model's" begin
        built = build_kernel(bound)
        layout = built.layout
        for shift in (0.0, 0.4)
            u = collect(range(-0.3 + shift, 0.5 - shift; length = layout.total))
            nt = constrain(layout, u)
            lik, prior = _oracle(nt)
            q = Base.invokelatest(prepare_query, built, bound, :likelihood)
            @test Base.invokelatest(q, u) ≈ lik rtol = 1e-12
            q = Base.invokelatest(prepare_query, built, bound, :prior)
            @test Base.invokelatest(q, u) ≈ prior rtol = 1e-12
            sq = prepare_sampler(built, bound, u; backend = BACKEND)
            grad = similar(u)
            value, _ = sampler_value_and_gradient!(sq, grad, u)
            @test value ≈ lik + prior + logjac(layout, u) rtol = 1e-12
            h = 1e-6
            fd = map(eachindex(u)) do i
                up = copy(u); up[i] += h
                dn = copy(u); dn[i] -= h
                (sq(up) - sq(dn)) / 2h
            end
            @test grad ≈ fd rtol = 1e-5 atol = 1e-7
        end
    end
end

@testset "generated names never take a name the model uses" begin
    taken = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1.0)
        y_scale = shiftby(x, 1.0)
        b_scale = 0.25
        b ~ Normal(0, 2 * s)
        y .~ Normal.(a .+ b .* y_scale .+ b_scale, shiftby(s, 0.5))
    end
    text = _text(_bound(taken, (; x, y)))
    @test occursin("y_scale_2 = shiftby(s, 0.5)", text)
    @test occursin("b_scale_2 = 2s", text)
    # A data column takes the name first too.
    data = (; x, y, y_scale = x)
    model = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1.0)
        y .~ Normal.(a .+ y_scale, shiftby(s, 0.5))
    end
    @test occursin("y_scale_2 = shiftby(s, 0.5)", _text(_bound(model, data)))
    # A synthesized response name (`y_eta`) is never a computed argument's.
    nested = quote
        a ~ Normal(0, 1)
        y .~ Normal.(a .+ shiftby(x .* a, 0.5), 1.0)
    end
    @test occursin("y_shiftby = shiftby(x .* a, 0.5)", _text(_bound(nested, (; x, y))))
end

@testset "a literal zero bound contributes nothing to its transform" begin
    model = quote
        s ~ restricted(Normal(0, 1), 0.0, Inf)
        d[1:2] .~ restricted.(Exponential.(0.125), 0.0, 0.5)
        n ~ restricted(Normal(0, 1), -Inf, 0)
        p ~ Uniform(0, 1)
        h ~ Uniform(-1, 2)
        y .~ Normal.(p .+ n .* x .+ h, s + d[1] + d[2])
    end
    bound = _bound(model, (; x, y))
    text = _text(bound)
    @test occursin("s::Float64 = exp(_ppl_fl_s)", text)
    @test occursin("_ppl_fl_s::Float64 = log(s)", text)
    @test occursin("d::AbstractVector{Float64} = 0.5 ./ (1 .+ exp.(-_ppl_int_d))", text)
    @test occursin("_ppl_int_d::AbstractVector{Float64} = log.(d) .- log.(0.5 .- d)", text)
    @test occursin("n::Float64 = -(exp(_ppl_up_n))", text)
    @test occursin("p::Float64 = 1.0 / (1 + exp(-_ppl_int_p))", text)
    @test occursin("h::Float64 = -1.0 + 3.0 / (1 + exp(-_ppl_int_h))", text)
    @test occursin("log(h + 1.0)", text)
    for gratuitous in ("0.0 +", "0.0 .+", "- 0.0", ".- 0.0", "0.0 -", "0.0 .-",
            "- -", ".- -", "log(1.0)")
        @test !occursin(gratuitous, text)
    end
    built = build_kernel(bound)
    layout = built.layout
    for shift in (0.0, 0.4)
        u = collect(range(-0.3 + shift, 0.5 - shift; length = layout.total))
        nt = constrain(layout, u)
        sd = nt.s + nt.d[1] + nt.d[2]
        lik = sum(logpdf.(Normal.(nt.p .+ nt.n .* x .+ nt.h, sd), y))
        prior = logpdf(Normal(0, 1), nt.s) + sum(logpdf.(Exponential(0.125), nt.d)) +
            logpdf(Normal(0, 1), nt.n) + logpdf(Uniform(0, 1), nt.p) +
            logpdf(Uniform(-1, 2), nt.h)
        sq = prepare_sampler(built, bound, u; backend = BACKEND)
        grad = similar(u)
        value, _ = sampler_value_and_gradient!(sq, grad, u)
        @test value ≈ lik + prior + logjac(layout, u) rtol = 1e-12
        h = 1e-6
        fd = map(eachindex(u)) do i
            up = copy(u); up[i] += h
            dn = copy(u); dn[i] -= h
            (sq(up) - sq(dn)) / 2h
        end
        @test grad ≈ fd rtol = 1e-5 atol = 1e-7
    end
end

@testset "promotion stays for values that may not be Float64" begin
    m = [1, 0]
    model = quote
        b[1:2] .~ Normal.(m, 1)
        y .~ Normal.(b[1] .+ b[2] .* x, 1.0)
    end
    text = _text(_bound(model, (; x, y, m)))
    @test occursin("plate(b, m) do b, m", text)
    @test occursin("(normal(1.0m, 1.0)).logpdf(b)", text)
end

# Module kernels composed into a model: the built program shows each call's
# values under the model's names, with no `identity` alias for a typed formal
# and no generated names.
module ComposedFns
using ReactiveKernels
@kernel rise(v) = begin
    xi = v .* 2.0
    value = xi .+ 1.0
    return value
end
@kernel fall(v, vm) = begin
    xi = v .- 3.0
    xi_max = vm .- 3.0
    value = xi .- xi_max
    return value
end
@kernel scaled_total(v::AbstractVector{Float64}, s) = begin
    total = sum(v) * s
    return total
end
end

@testset "composed kernels in the built program read under the model's names" begin
    model = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sa ~ Exponential(4.0)
        sp ~ Exponential(4.0)
        base = a .+ b .* x
        mu = base .* ComposedFns.rise(x) .* exp.(ComposedFns.fall(x, 0.5))
        shift = ComposedFns.scaled_total(base, 0.1)
        y .~ Normal.(mu .+ shift, sa + sp)
    end
    data = (; x, y)
    bound = _bound(model, data; mod = @__MODULE__)
    built = build_kernel(bound)
    code = string(readable_code(built.spec))
    # A call inside an expression is named under its assignment; a typed
    # formal reads the model's value with no `identity` alias.
    @test occursin(r"var\"[^\"]+\.fall\.xi_max\" = 0\.5 \.- 3\.0", code)
    @test occursin("shift = sum(base) * 0.1", code)
    @test !occursin("identity(", code)
    @test !occursin("##", code)
    @test !occursin(r"\b(xi|value)_\d+\b", code)

    oracle(nt) = begin
        base = nt.a .+ nt.b .* x
        mu = base .* (x .* 2.0 .+ 1.0) .* exp.((x .- 3.0) .- (0.5 - 3.0))
        sum(logpdf.(Normal.(mu .+ sum(base) * 0.1, nt.sa + nt.sp), y)) +
            logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
            logpdf(Exponential(4.0), nt.sa) + logpdf(Exponential(4.0), nt.sp)
    end
    layout = built.layout
    u = [0.2, -0.3, 0.1, -0.4]
    sq = prepare_sampler(built, bound, u; backend = BACKEND)
    grad = similar(u)
    value, _ = sampler_value_and_gradient!(sq, grad, u)
    @test value ≈ oracle(constrain(layout, u)) + logjac(layout, u) rtol = 1e-12
    h = 1e-6
    fd = map(eachindex(u)) do i
        up = copy(u); up[i] += h
        dn = copy(u); dn[i] -= h
        (sq(up) - sq(dn)) / 2h
    end
    @test grad ≈ fd rtol = 1e-5 atol = 1e-7
end

# A definition with no coefficient structure is one value, read under its
# own name like any retained definition: the built program shows `mu`, and
# the calls lifted out of it are scoped under `mu`, never under a column
# the lowering extracted and named itself (snag `read-the-built-p-bd95fad7`).
@testset "a definition without coefficient structure keeps its name" begin
    rise(v) = v .* 2.0 .+ 1.0
    fall(v) = (v .- 3.0) .- (0.5 - 3.0)
    head = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sa ~ Exponential(1.0)
        base = a .+ b .* x
    end
    head_lp(nt) = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Exponential(1.0), nt.sa)
    base(nt) = nt.a .+ nt.b .* x
    yc = [1, 0, 2, 1]
    for (body, wants, oracle) in (
            # The reported spelling: module calls inline in the location.
            (quote
                mu = base .* ComposedFns.rise(base) .* exp.(ComposedFns.fall(base, 0.5))
                y .~ Normal.(mu, sa)
            end, ["var\"mu.rise.xi\" = base .* 2.0",
                  "mu = (base .* var\"mu.rise\") .* exp.(var\"mu.fall\")",
                  "plate(y, mu, sa) do y, mu, sa"],
                nt -> sum(logpdf.(Normal.(base(nt) .* rise(base(nt)) .*
                    exp.(fall(base(nt))), nt.sa), y))),
            # Its named twin.
            (quote
                r = ComposedFns.rise(base)
                f = ComposedFns.fall(base, 0.5)
                mu = base .* r .* exp.(f)
                y .~ Normal.(mu, sa)
            end, ["mu = (base .* r) .* exp.(f)", "plate(y, mu, sa) do y, mu, sa"],
                nt -> sum(logpdf.(Normal.(base(nt) .* rise(base(nt)) .*
                    exp.(fall(base(nt))), nt.sa), y))),
            # A scale, a log-link location and a location two responses read.
            (quote
                r = ComposedFns.rise(base)
                sd = exp.(0.1 .* base .* r)
                y .~ Normal.(a, sd)
            end, ["sd = exp.((0.1 .* base) .* r)", ", sd) do y, "],
                nt -> sum(logpdf.(Normal.(nt.a,
                    exp.(0.1 .* base(nt) .* rise(base(nt)))), y))),
            (quote
                r = ComposedFns.rise(base)
                eta = 0.1 .* base .* r
                yc .~ Poisson.(exp.(eta))
            end, ["eta = (0.1 .* base) .* r", "dot(_ppl_yf_yc_resp, eta)"],
                nt -> sum(logpdf.(Poisson.(exp.(0.1 .* base(nt) .*
                    rise(base(nt)))), yc))),
            (quote
                r = ComposedFns.rise(base)
                mu = base .* r
                y .~ Normal.(mu, sa)
                yc .~ Normal.(mu, 1.0)
            end, ["mu = base .* r", "plate(y, mu, sa) do y, mu, sa",
                  "plate(yc, mu) do yc, mu"],
                nt -> sum(logpdf.(Normal.(base(nt) .* rise(base(nt)), nt.sa), y)) +
                    sum(logpdf.(Normal.(base(nt) .* rise(base(nt)), 1.0), yc))))
        model = Expr(:block, head.args..., body.args...)
        data = (; x, y, yc)
        bound = bind_data(lower_rkppl(model, data; conditioned = (:y, :yc),
            mod = @__MODULE__), data)
        built = build_kernel(bound)
        code = string(readable_code(built.spec))
        for want in wants
            @test occursin(want, code)
        end
        @test !occursin("_rkppl_", code)
        layout = built.layout
        u = collect(range(-0.3, 0.4; length = layout.total))
        sq = prepare_sampler(built, bound, u; backend = BACKEND)
        grad = similar(u)
        value, _ = sampler_value_and_gradient!(sq, grad, u)
        nt = constrain(layout, u)
        @test value ≈ oracle(nt) + head_lp(nt) + logjac(layout, u) rtol = 1e-12
        h = 1e-6
        fd = map(eachindex(u)) do i
            up = copy(u); up[i] += h
            dn = copy(u); dn[i] -= h
            (sq(up) - sq(dn)) / 2h
        end
        @test grad ≈ fd rtol = 1e-5 atol = 1e-7
    end
end

@testset "a scalar parameter's log-Jacobian term is named after it" begin
    model = quote
        p ~ Beta(2.0, 2.0)
        q ~ Beta(2.0, 3.0)
        sigma ~ Exponential(1.0)
        y .~ Normal.(p + q, sigma)
    end
    bound = _bound(model, (; y); mod = @__MODULE__)
    built = build_kernel(bound)
    code = string(readable_code(built.spec))
    @test occursin("var\"p.logjac\" = log(var\"p.logjac.logjac__x\")", code)
    # The exp transform's term is the coordinate itself, read in place.
    @test occursin("log_jacobian = (var\"p.logjac\" + var\"q.logjac\") + unconstrained[3]",
                   code)
    @test !occursin("##", code)
    @test !occursin("let ", code)
    layout = built.layout
    u = [0.3, -0.2, 0.1]
    query = prepare_query(built, bound, :log_jacobian)
    @test Base.invokelatest(query, u) ≈ logjac(layout, u) rtol = 1e-12
end

end
