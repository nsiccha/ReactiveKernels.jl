module QueryInputsTests
using ReactiveKernels, ReactiveKernelsPPL, Test
import Distributions

# Runtime inputs (snag rkppl-prepare-qu-753cb7c9; design decision 0jt3vu3,
# option 1): `prepare_query(built, plan, want; inputs = (:x,))` prepares once
# and is called as `q(u, xx)`, giving what a binding with `x = xx` gives.
# Everything a binding computes from `x` is evaluated again per call; what a
# call cannot check or reproduce is refused, naming why.

_qi(q, args...) = Base.invokelatest(q, args...)
_rebound(built, plan, want, u) =
    Base.invokelatest(Base.invokelatest(prepare_query, built, plan, want), u)

_refused(f, needle) = try
    f()
    false
catch e
    e isa ContractValidationError && occursin(needle, sprint(showerror, e))
end
_mismatch(f, needle) = try
    f()
    false
catch e
    e isa ArgumentError && occursin(needle, sprint(showerror, e))
end

dbl(v) = 2 .* v
nbig(v) = count(>(0.5), v)

# The reporter's toy: a recursion over data `x`, the response absent.
const QI_SCAN = @rkppl begin
    rho ~ LogNormal(0.0, 1.0)
    level ~ Normal(0.0, 1.0)
    @scan begin
        state[1] = x[1]
        for t in 2:n
            state[t] = exp(-rho) * state[t - 1] + x[t]
        end
    end
    background = level .+ state
    m = background .+ 0.0
    y .~ Normal.(m, 1.0)
end

@testset "a scan over runtime data matches rebinding, bit for bit" begin
    n = 40
    design = QI_SCAN(; n, x = rand(n))
    built = build_kernel(design)
    u = randn(built.layout.total)
    q = prepare_query(built, design, (:background,); inputs = (:x,))
    both = prepare_query(built, design, (:background, :state); inputs = :x)
    for _ in 1:5
        xx = rand(n)
        expected = _rebound(built, QI_SCAN(; n, x = xx), (:background,), u)
        @test _qi(q, u, xx) == expected
        @test _qi(both, u, xx) ==
            _rebound(built, QI_SCAN(; n, x = xx), (:background, :state), u)
    end
    @test occursin("q(u, x)", sprint(show, q))
end

const QI_DEFINED = @rkppl begin
    level ~ Normal(0.0, 1.0)
    w = dbl(x)
    background = level .+ w
    y .~ Normal.(background, 1.0)
end

const QI_INLINE = @rkppl begin
    level ~ Normal(0.0, 1.0)
    background = level .+ dbl(x)
    y .~ Normal.(background, 1.0)
end

@testset "what binding computes from an input follows the call" begin
    x0, x1 = collect(1.0:5.0), collect(10.0:14.0)
    for model in (QI_DEFINED, QI_INLINE)
        design = model(; x = x0)
        built = build_kernel(design)
        u = [0.25]
        q = prepare_query(built, design, (:background,); inputs = (:x,))
        @test _qi(q, u, x1) == _rebound(built, model(; x = x1), (:background,), u)
        @test _qi(q, u, x1) == 0.25 .+ 2 .* x1
        # The binding's own values reproduce the bound query.
        @test _qi(q, u, x0) == _rebound(built, design, (:background,), u)
    end
end

const QI_SCALAR = @rkppl begin
    level ~ Normal(0.0, 1.0)
    background = level .+ s .* x
    y .~ Normal.(background, 1.0)
end

@testset "a bound number is named as authored" begin
    x0, x1 = collect(1.0:5.0), collect(10.0:14.0)
    design = QI_SCALAR(; x = x0, s = 2.0)
    built = build_kernel(design)
    u = [-0.5]
    q = prepare_query(built, design, (:background,); inputs = (:s, :x))
    @test _qi(q, u, 3.0, x1) ==
        _rebound(built, QI_SCALAR(; x = x1, s = 3.0), (:background,), u)
    qs = prepare_query(built, design, (:background,); inputs = (:s,))
    @test _qi(qs, u, 3.0) == -0.5 .+ 3.0 .* x0
end

@testset "presets take runtime inputs; observations stay bound" begin
    x0, x1 = collect(1.0:5.0), collect(10.0:14.0)
    yv = collect(21.0:25.0)
    design = QI_DEFINED(; x = x0) | (; y = yv)
    built = build_kernel(design)
    u = [0.75]
    observed(xx) = QI_DEFINED(; x = xx) | (; y = yv)
    for preset in (:likelihood, :pointwise, :observations, :sampler)
        q = prepare_query(built, design, preset; inputs = (:x,))
        @test _qi(q, u, x1) == _rebound(built, observed(x1), preset, u)
    end
    # The likelihood observes the bound `y` at the call's `x`.
    q = prepare_query(built, design, :likelihood; inputs = (:x,))
    @test _qi(q, u, x1) ≈
        sum(Distributions.logpdf.(Distributions.Normal.(0.75 .+ 2 .* x1, 1.0), yv))
    # Exemplars of the parameter vector cover the inputs from the binding.
    q = prepare_query(built, design, :likelihood; inputs = (:x,), exemplars = (u,))
    @test _qi(q, u, x1) == _rebound(built, observed(x1), :likelihood, u)
end

const QI_MATRIX = @rkppl begin
    X = hcat(ones(length(x)), x)
    b[axes(X, 2)] .~ Normal.(0, 2)
    mu = X * b
    y .~ Normal.(mu, 1.0)
end

@testset "a value matrix over an input follows the call" begin
    x0, x1 = [-1.0, 0.5, 2.0], [3.0, -2.0, 0.25]
    design = QI_MATRIX(; x = x0)
    built = build_kernel(design)
    u = [0.2, -0.3]
    q = prepare_query(built, design, (:mu,); inputs = (:x,))
    @test _qi(q, u, x1) == _rebound(built, QI_MATRIX(; x = x1), (:mu,), u)
    @test _qi(q, u, x1) ≈ 0.2 .- 0.3 .* x1
end

const QI_COUNTED = @rkppl begin
    level ~ Normal(0.0, 1.0)
    k = nbig(x)
    background = level .+ k .* x
    y .~ Normal.(background, 1.0)
end

@testset "per-call checks" begin
    x0 = [0.1, 0.9, 0.2]
    design = QI_COUNTED(; x = x0)
    built = build_kernel(design)
    u = [0.0]
    q = prepare_query(built, design, (:background,); inputs = (:x,))
    # Floating-point values may change ...
    @test _qi(q, u, [0.2, 0.8, 0.3]) == 1 .* [0.2, 0.8, 0.3]
    # ... but an integer binding computes from them may not: it may index,
    # count or size the program the preparation holds.
    @test _mismatch(() -> _qi(q, u, [0.9, 0.8, 0.3]), "`k`")
    # The program and its extents are the binding's: type and size stay.
    @test _mismatch(() -> _qi(q, u, [0.1, 0.2]), "size")
    @test _mismatch(() -> _qi(q, u, Float32[0.1, 0.2, 0.3]), "type")
    @test _mismatch(() -> _qi(q, u), "1 runtime input")
end

const QI_FACTOR = @rkppl begin
    c[levels(g)] .~ Normal.(0, 2)
    mu = c[g] .+ x
    y .~ Normal.(mu, 1.0)
end

const QI_DERIVED_OBSERVATION = @rkppl begin
    level ~ Normal(0.0, 1.0)
    z = dbl(x)
    z .~ Normal.(level, 1.0)
end

@testset "refused inputs name why" begin
    x0 = collect(1.0:5.0)
    design = QI_DEFINED(; x = x0) | (; y = rand(5))
    built = build_kernel(design)
    # Only data values the binding was given can be supplied.
    @test _refused(() -> prepare_query(built, design, :likelihood; inputs = (:z,)),
        "has no data value `z`")
    @test _refused(() -> prepare_query(built, design, :likelihood; inputs = (:w,)),
        "computed by binding")
    @test _refused(() -> prepare_query(built, design, :likelihood; inputs = (:x, :x)),
        "twice")
    # Binding checks observations (support, missing entries); a call would
    # not. That covers an observation binding computes from the input.
    @test _refused(() -> prepare_query(built, design, :likelihood; inputs = (:y,)),
        "as an observation")
    derived = QI_DERIVED_OBSERVATION(; x = x0)
    @test _refused(() -> prepare_query(build_kernel(derived), derived, :likelihood;
        inputs = (:x,)), "computes `z`")
    # A value pinned in the call is data, not an observation.
    pinned = QI_DEFINED(; x = x0, y = rand(5))
    qp = prepare_query(build_kernel(pinned), pinned, (:background,); inputs = (:y,))
    @test _qi(qp, [0.0], rand(5)) == 2 .* x0
    # Integer data index, count and size the program; binding checks them.
    scan = QI_SCAN(; n = 4, x = rand(4))
    @test _refused(() -> prepare_query(build_kernel(scan), scan, (:background,);
        inputs = (:n,)), "floating-point")
    # Factor labels map to level codes at binding.
    factor = QI_FACTOR(; g = [2.0, 1.0, 2.0], x = [0.1, 0.2, 0.3])
    @test _refused(() -> prepare_query(build_kernel(factor), factor, (:mu,);
        inputs = (:g,)), "level codes")
    q = prepare_query(build_kernel(factor), factor, (:mu,); inputs = (:x,))
    @test length(_qi(q, zeros(2), [1.0, 2.0, 3.0])) == 3
end

end # module
