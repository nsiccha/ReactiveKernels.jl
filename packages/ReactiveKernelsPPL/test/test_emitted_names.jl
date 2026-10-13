module EmittedNameTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test

# The generated pre-build definition reads with the model's own names
# (rkppl-use §16):
# plate cells name their arguments after the values they iterate, a value
# the lowering computes for a distribution argument is named after its
# owner and role, a declared array's prior is named after the array, sums
# carry no `0.0` seed, `Float64` values need no `1.0 *` port and the
# emitted source qualifies functions from the model module. Every density,
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

_text(bound) = sprint(print,
    Base.remove_linenums!(deepcopy(kernel_expr(bound, assign_layout(bound)))))
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
        # `loc` is a retained value (`spread` reads it too), and the
        # response reads that graph value without a synthetic alias.
        @test occursin("plate(y, loc, y_scale) do y, loc, y_scale", text)
        @test occursin("(normal(loc, y_scale)).logpdf(y)", text)
        @test !occursin("y_eta = loc", text)
        @test !occursin("_ppl_lp_y_eta", text)
        @test occursin("_ppl_pw_y_resp = (__ops__[",
            string(code_expr(build_kernel(bound).spec)))
        @test !occursin(r"_ppl_c\d", text)
    end

    @testset "a computed distribution argument is named after its role" begin
        @test occursin("y_scale = Main.EmittedNameTests.Fns.spread(loc, sa, sp)", text)
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

    @testset "the generated expression keeps its global references" begin
        code = kernel_expr(bound, assign_layout(bound))
        @test occursin("Fns.spread", sprint(print, code))
        @test occursin("Fns.spread", text)
    end

    @testset "values and native gradients are the model's" begin
        built = build_kernel(bound)
        layout = built.layout
        # Compare the emitted graph with the unelided `y_eta = loc` graph.
        prior_def = Base.invokelatest(ReactiveKernelsPPL._kernel_expr_latest,
            bound, layout)
        prior_def = ReactiveKernelsPPL._name_plate_cells(prior_def,
            ReactiveKernelsPPL._location_value_aliases(bound, prior_def))
        prior_built = (; spec = Base.invokelatest(
            ReactiveKernelsPPL._eval_kernel_def, prior_def), layout,
            program = built.program)
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
            prior_sq = prepare_sampler(prior_built, bound, u; backend = BACKEND)
            prior_grad = similar(u)
            prior_value, _ = sampler_value_and_gradient!(prior_sq, prior_grad, u)
            @test isequal(value, prior_value)
            @test isequal(grad, prior_grad)
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

@testset "a bare named opaque location reads its definition" begin
    model = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        loc = shiftby(x, a)
        y .~ Normal.(loc, s)
    end
    plate_counts = Int[]
    for n in (1, 2)
        bound = _bound(model, (; x = repeat(x, n), y = repeat(y, n)))
        text = _text(bound)
        @test occursin("loc = Main.EmittedNameTests.Fns.shiftby(x, a)", text)
        @test occursin("plate(y, ReactiveKernelsPPL._ppl_rows(loc, ReactiveKernelsPPL._observation_rows(y)), s) do y, loc, s", text)
        @test !occursin("y_eta =", text)
        @test !occursin("_ppl_lp_y_eta", text)
        built = build_kernel(bound)
        push!(plate_counts, count(item -> item.kind === :plate,
            recipe_inventory(built.spec)))
        if n == 1
            layout = built.layout
            prior_def = Base.invokelatest(ReactiveKernelsPPL._kernel_expr_latest,
                bound, layout)
            prior_built = (; spec = Base.invokelatest(
                ReactiveKernelsPPL._eval_kernel_def, prior_def), layout,
                program = built.program)
            u = [0.2, 0.1]
            current = prepare_sampler(built, bound, u; backend = BACKEND)
            prior = prepare_sampler(prior_built, bound, u; backend = BACKEND)
            current_grad, prior_grad = similar(u), similar(u)
            current_value, _ = sampler_value_and_gradient!(current, current_grad, u)
            prior_value, _ = sampler_value_and_gradient!(prior, prior_grad, u)
            @test isequal(current_value, prior_value)
            @test isequal(current_grad, prior_grad)
        end
    end
    @test plate_counts[1] >= 1
    @test plate_counts[1] == plate_counts[2]
end

@testset "an elided location alias leaves same-named cell parameters alone" begin
    # The scalar priors' cell binds `value, location, scale`; eliding the
    # authored `location = value` must not rename those parameters.
    y3 = y[1:3]
    model = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        value = [a, b, a + b]
        location = value
        y .~ Normal.(location, 1.0)
    end
    bound = _bound(model, (; y = y3))
    text = _text(bound)
    @test !occursin("location = value", text)
    @test occursin("do value, location, scale", text)
    @test occursin("(normal(location, scale)).logpdf(value)", text)
    built = build_kernel(bound)
    @test coordinate_names(built.layout) == [:a, :b]
    u = [0.2, -0.3]
    mean(q) = [q[1], q[2], q[1] + q[2]]
    oracle(q) = sum(logpdf.(Normal.(mean(q), 1.0), y3)) +
        logpdf(Normal(0, 1), q[1]) + logpdf(Normal(0, 1), q[2])
    r = y3 .- mean(u)
    grad = similar(u)
    value, _ = sampler_value_and_gradient!(
        prepare_sampler(built, bound, u; backend = BACKEND), grad, u)
    @test value ≈ oracle(u) rtol = 1e-12
    @test grad ≈ [r[1] + r[3] - u[1], r[2] + r[3] - u[2]] rtol = 1e-10

    subst(ex, map) = Base.remove_linenums!(ReactiveKernelsPPL._hsubst(
        ex, Dict{Symbol,Any}(map); shadow = true))
    @test subst(:(plate(location) do location; location end), :location => :mu) ==
        Base.remove_linenums!(:(plate(mu) do location; location end))
    @test subst(:(plate(v) do value; location + value end), :location => :mu) ==
        Base.remove_linenums!(:(plate(v) do value; mu + value end))
    # refused: the substituted value would read the cell's own parameter
    # instead of the outer value it names (dev §1, never silently wrong).
    @test_throws ArgumentError subst(:(plate(v) do value; location + value end),
        :location => :value)
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
    @test occursin("y_scale_2 = Main.EmittedNameTests.Fns.shiftby(s, 0.5)", text)
    @test occursin("b_scale_2 = 2s", text)
    # A data column takes the name first too.
    data = (; x, y, y_scale = x)
    model = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1.0)
        y .~ Normal.(a .+ y_scale, shiftby(s, 0.5))
    end
    @test occursin("y_scale_2 = Main.EmittedNameTests.Fns.shiftby(s, 0.5)", _text(_bound(model, data)))
    # A synthesized response name (`y_eta`) is never a computed argument's.
    nested = quote
        a ~ Normal(0, 1)
        y .~ Normal.(a .+ shiftby(x .* a, 0.5), 1.0)
    end
    @test occursin("y_shiftby = Main.EmittedNameTests.Fns.shiftby(x .* a, 0.5)", _text(_bound(nested, (; x, y))))
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
    @test occursin("d::AbstractVector{Float64} = rk_affine_logistic.(0.0, 0.5, _ppl_int_d)", text)
    @test occursin("_ppl_int_d::AbstractVector{Float64} = log.(d) .- log.(0.5 .- d)", text)
    @test occursin("n::Float64 = -(exp(_ppl_up_n))", text)
    @test occursin("p::Float64 = rk_affine_logistic(0.0, 1.0, _ppl_int_p)", text)
    @test occursin("h::Float64 = rk_affine_logistic(-1.0, 2.0, _ppl_int_h)", text)
    @test occursin("log(h + 1.0)", text)
    # The separate logistic log-Jacobian still emits `0.0 - log1pexp(...)`;
    # this test concerns the literal bounds in the constrained transforms.
    for gratuitous in ("0.0 +", "0.0 .+", "- 0.0", ".- 0.0", "0.0 .-",
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

@testset "composed kernels keep their generated operation slots" begin
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
    code = string(code_expr(built.spec))
    # The planned expression retains the call slots and their actual bindings.
    # Source bodies of those callables are separate from this expression.
    @test occursin(r"var\"[^\"]+\.fall\.xi_max\" = \(__ops__\[", code)
    @test occursin(r"shift = \(__ops__\[\d+\]\)\(base, var\"shift\.s\"\)", code)
    @test !occursin("identity(", code)
    @test occursin("var\"##plate_argument#", code)

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
# own name like any retained definition: the planned expression binds `mu`, and
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
            end, ["var\"mu.rise.xi\" = (__ops__[", "mu = (__ops__["],
                nt -> sum(logpdf.(Normal.(base(nt) .* rise(base(nt)) .*
                    exp.(fall(base(nt))), nt.sa), y))),
            # Its named twin.
            (quote
                r = ComposedFns.rise(base)
                f = ComposedFns.fall(base, 0.5)
                mu = base .* r .* exp.(f)
                y .~ Normal.(mu, sa)
            end, ["r = (__ops__[", "f = (__ops__[", "mu = (__ops__["],
                nt -> sum(logpdf.(Normal.(base(nt) .* rise(base(nt)) .*
                    exp.(fall(base(nt))), nt.sa), y))),
            # A scale, a log-link location and a location two responses read.
            (quote
                r = ComposedFns.rise(base)
                sd = exp.(0.1 .* base .* r)
                y .~ Normal.(a, sd)
            end, ["sd = (__ops__["],
                nt -> sum(logpdf.(Normal.(nt.a,
                    exp.(0.1 .* base(nt) .* rise(base(nt)))), y))),
            (quote
                r = ComposedFns.rise(base)
                eta = 0.1 .* base .* r
                yc .~ Poisson.(exp.(eta))
            end, ["eta = (__ops__["],
                nt -> sum(logpdf.(Poisson.(exp.(0.1 .* base(nt) .*
                    rise(base(nt)))), yc))),
            (quote
                r = ComposedFns.rise(base)
                mu = base .* r
                y .~ Normal.(mu, sa)
                yc .~ Normal.(mu, 1.0)
            end, ["mu = (__ops__["],
                nt -> sum(logpdf.(Normal.(base(nt) .* rise(base(nt)), nt.sa), y)) +
                    sum(logpdf.(Normal.(base(nt) .* rise(base(nt)), 1.0), yc))))
        model = Expr(:block, head.args..., body.args...)
        data = (; x, y, yc)
        bound = bind_data(lower_rkppl(model, data; conditioned = (:y, :yc),
            mod = @__MODULE__), data)
        built = build_kernel(bound)
        code = string(code_expr(built.spec))
        for want in wants
            @test occursin(want, code)
        end
        @test occursin("__ops__", code)
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
    code = string(code_expr(built.spec))
    @test occursin("var\"p.logjac\" = (__ops__[", code)
    @test occursin("log_jacobian = (__ops__[", code)
    layout = built.layout
    u = [0.3, -0.2, 0.1]
    query = prepare_query(built, bound, :log_jacobian)
    @test Base.invokelatest(query, u) ≈ logjac(layout, u) rtol = 1e-12
end

end
