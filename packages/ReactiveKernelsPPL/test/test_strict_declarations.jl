# Strict declarations (user decision 05oe96l): no parameter is ever minted
# from an undeclared name. A coefficient name that is not data, a
# definition, or a declared parameter fails at lowering, naming the
# declaration to write, on both entry points (`@rkppl` and direct
# `lower_rkppl`, the BRM emitter's path). Declaring the formerly implied
# `Normal(0, 1)` lowers to the identical plan (pinned by the corpus guard).
using ReactiveKernelsPPL
using Test

function _strict_err(f)
    try
        f()
        return nothing
    catch e
        return e
    end
end

const _STRICT_COLS = (; y = [1.0, 2.0, 1.5, 2.5], x = [0.5, -1.0, 1.5, 0.0],
    x1 = [0.5, -1.0, 1.5, 0.0], x2 = [1.0, 0.5, -0.5, 2.0])

@testset "strict declarations: undeclared scalar coefficient" begin
    # `aa` is a typo for a declared name, or simply undeclared.
    body = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = aa .+ b .* x
        y .~ Normal.(mu, 1.0)
    end
    err = _strict_err(() -> lower_rkppl(body, (:y, :x); conditioned = (:y, :x)))
    # refused: every coefficient needs a declaration (P6, 05oe96l).
    @test err isa SurfaceLoweringError
    msg = sprint(showerror, err)
    @test occursin("`aa` is not a data column, a definition, or a declared " *
                   "parameter", msg)
    @test occursin("`aa ~ Normal(0, 1)`", msg)
    # The macro entry point is equally strict.
    m = @rkppl begin
        b ~ Normal(0, 1)
        mu = aa .+ b .* x
        y .~ Normal.(mu, 1.0)
    end
    # refused: the macro also refuses undeclared aa (P6, 05oe96l).
    @test _strict_err(() -> (m(; x = _STRICT_COLS.x) | (; y = _STRICT_COLS.y))) isa
        SurfaceLoweringError
    # Declared, it lowers.
    ok = lower_rkppl(quote
            aa ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = aa .+ b .* x
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    @test isempty(ok.population_priors)
    @test Set(p.name for p in ok.parameters) == Set((:aa, :b))
end

@testset "strict declarations: undeclared matrix coefficient vector" begin
    err = _strict_err(() -> lower_rkppl(quote
            X = hcat(ones(length(x1)), x1, x2)
            mu = X * b
            y .~ Normal.(mu, 1.0)
        end, (:y, :x1, :x2); conditioned = (:y, :x1, :x2)))
    # refused: every coefficient needs a declaration (P6, 05oe96l).
    @test err isa SurfaceLoweringError
    @test occursin("`b[axes(X, 2)] .~ Normal.(0, 1)`", sprint(showerror, err))
end

@testset "strict declarations: undeclared GLM-object beta" begin
    err = _strict_err(() -> lower_rkppl(quote
            X = hcat(x1, x2)
            alpha ~ Normal(0, 10)
            y ~ NormalIDGLM(X, alpha, beta, 1.0)
        end, (:y, :x1, :x2); conditioned = (:y, :x1, :x2)))
    # refused: every coefficient needs a declaration (P6, 05oe96l).
    @test err isa SurfaceLoweringError
    @test occursin("`beta[axes(X, 2)] .~ Normal.(0, 1)`", sprint(showerror, err))
end

@testset "strict declarations: computed coefficient predictor" begin
    err = _strict_err(() -> lower_rkppl(quote
            raw ~ Normal(0, 1)
            lambda ~ HalfCauchy(1)
            tau ~ HalfCauchy(1)
            b1 = raw * lambda * tau
            mu = a .+ b1 .* x1
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, (:y, :x1); conditioned = (:y, :x1)))
    # refused: every coefficient needs a declaration (P6, 05oe96l).
    @test err isa SurfaceLoweringError
    msg = sprint(showerror, err)
    @test occursin("`a ~ Normal(0, 1)`", msg)
end
