using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels,
    ReactiveKernelsPPL, Test

@rkppl _ntv_bounds(x, scale) = begin
    a ~ Normal(0.0, scale)
    b ~ Normal(1.0, 0.2)
    r = a .+ b .* x
    return (lo = r, hi = a .- b .* x)
end

@rkppl _ntv_keywords(x, scale, extra) = begin
    a ~ Normal(0.0, scale)
    b ~ Normal(1.0, 0.2)
    lo = a .+ b .* x
    return (; lo, hi = a .- b .* x, extra)
end

@rkppl _ntv_nested(x, scale, extra) = begin
    a ~ Normal(0.0, scale)
    b ~ Normal(1.0, 0.2)
    r = a .+ b .* x
    return ((r, a .- b .* x), extra)
end

const _NTV_MODELS = (
    control = @rkppl(begin
        scale ~ Exponential(1.0)
        k ~ _ntv_bounds(x, scale)
        rec = getproperty(k, :lo)
        y1 .~ Normal.(rec, 0.5)
        y2 .~ Normal.(getproperty(k, :hi), 0.5)
    end),
    label = @rkppl(begin
        scale ~ Exponential(1.0)
        k ~ _ntv_bounds(x, scale)
        lo = getproperty(k, :lo)
        y1 .~ Normal.(lo, 0.5)
        y2 .~ Normal.(getproperty(k, :hi), 0.5)
    end),
    mixed = @rkppl(begin
        scale ~ Exponential(1.0)
        k ~ _ntv_bounds(x, scale)
        y1 .~ Normal.(getproperty(k, :lo), 0.5)
        y2 .~ Normal.(getproperty(k, :lo) .+ getproperty(k, :hi), 0.5)
    end),
    mixed_reverse = @rkppl(begin
        scale ~ Exponential(1.0)
        k ~ _ntv_bounds(x, scale)
        y2 .~ Normal.(getproperty(k, :lo) .+ getproperty(k, :hi), 0.5)
        y1 .~ Normal.(getproperty(k, :lo), 0.5)
    end),
    unread = @rkppl(begin
        scale ~ Exponential(1.0)
        k ~ _ntv_bounds(x, scale)
        v = 2 .* x
        y1 .~ Normal.(v, 0.5)
        y2 .~ Normal.(v .+ scale, 0.5)
    end),
    locals = @rkppl(begin
        scale ~ Exponential(1.0)
        k ~ _ntv_bounds(x, scale)
        y1 .~ Normal.(k.r, 0.5)
        y2 .~ Normal.(k.r .+ k.r, 0.5)
    end),
    keywords = @rkppl(begin
        scale ~ Exponential(1.0)
        k ~ _ntv_keywords(x, scale, extra)
        lo = getproperty(k, :lo)
        y1 .~ Normal.(lo, 0.5)
        y2 .~ Normal.(getproperty(k, :hi), 0.5)
    end),
    nested = @rkppl(begin
        scale ~ Exponential(1.0)
        k ~ _ntv_nested(x, scale, extra)
        y1 .~ Normal.(getindex(getindex(k, 1), 1), 0.5)
        y2 .~ Normal.(getindex(getindex(k, 1), 2), 0.5)
    end),
)

function _ntv_reference(kind, u, x, y1, y2)
    scale, a, b = exp(u[1]), u[2], u[3]
    lo, hi = a .+ b .* x, a .- b .* x
    m1, m2 = kind in (:mixed, :mixed_reverse) ? (lo, lo .+ hi) :
        kind === :unread ? (2 .* x, 2 .* x .+ scale) :
        kind === :locals ? (lo, 2 .* lo) : (lo, hi)
    return logpdf(Exponential(1.0), scale) + u[1] +
        logpdf(Normal(0.0, scale), a) + logpdf(Normal(1.0, 0.2), b) +
        sum(logpdf.(Normal.(m1, 0.5), y1)) +
        sum(logpdf.(Normal.(m2, 0.5), y2))
end

function _ntv_fd(f, u; h = 1e-6)
    map(eachindex(u)) do i
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end

@testset "submodel named-tuple vector returns" begin
    programs = Dict{Symbol,Expr}()
    for n in (0, 3, 11), (kind, model) in pairs(_NTV_MODELS)
        @testset "$kind n=$n" begin
            x = [sin(0.7i) for i in 1:n]
            y1, y2 = cos.(x), sin.(x)
            extra = [4.0, 8.0] # A tuple field has its own axes.
            original = deepcopy((x, y1, y2, extra))
            inputs = kind in (:keywords, :nested) ? (; x, extra) : (; x)
            plan = model(; inputs...) | (; y1, y2)
            built = build_kernel(plan)
            if n == 3
                programs[kind] = built.program
            elseif n == 11
                # More rows must not expand the generated program.
                @test ReactiveKernelsPPL._same_program(programs[kind], built.program)
            end
            @test coordinate_names(built.layout) == [:scale, Symbol("k.a"), Symbol("k.b")]
            u = [0.1, -0.2, 0.7]
            sampler = prepare_sampler(built, plan, u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            reference(v) = _ntv_reference(kind, v, x, y1, y2)
            for v in (u, [-0.3, 0.4, 1.2])
                g = similar(v)
                actual, _ = sampler_value_and_gradient!(sampler, g, v)
                @test actual ≈ reference(v) rtol = 1e-12
                @test g ≈ _ntv_fd(reference, v) rtol = 1e-5 atol = 1e-7
            end
            @test (x, y1, y2, extra) == original
        end
    end
end
