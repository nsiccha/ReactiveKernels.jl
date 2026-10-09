module AuthoredNameTests
using Distributions, ReactiveKernels, ReactiveKernelsPPL, Test
import ReactiveKernelsPPL: ContractValidationError

# A definition the lowering optimizes into a response location, a scale or a
# composition's sub-predictor, or absorbs as a pure alias, is still a graph
# value under the name its author gave it (user decision `0fbe312`; "all
# intermediate quantities" are part of the RK graph). A predictor's node is
# named after the predictor (user decision `1c0jiwz`); a pure alias is an RK
# alias of the node it names: density, coordinates and recipes are those of
# the same plan without the alias.

const Y = [-0.4, 0.2, 0.7, -0.1]
const X = [0.1, -0.3, 0.5, 1.2]
const G = [1, 2, 1, 2]
const COUNTS = [1, 0, 3, 2]

query(built, bound, name, u) = Base.invokelatest(ReactiveKernels.prepare(built.spec;
    have = built.spec.have_names, want = (name,),
    bound = NamedTuple(k => v for (k, v) in bound.columns)), u)
density(built, bound, u) = Base.invokelatest(prepare_query(built, bound, :sampler), u)
recipes(spec) = length(recipe_inventory(spec))

# `values(nt)` gives the authored value of each queried name from the
# constrained parameters `nt` (a vector location reads the same value on
# every row it broadcasts to).
const CASES = [
    (label = "intercept alias", data = (;), obs = (; y = Y),
        model = @rkppl(begin
            a ~ Normal(0, 1); s ~ Exponential(1)
            eta = a
            y .~ Normal.(eta, s)
        end), values = nt -> (; eta = nt.a)),
    (label = "signed intercept alias", data = (;), obs = (; y = Y),
        model = @rkppl(begin
            a ~ Normal(0, 1); s ~ Exponential(1)
            eta = -a
            y .~ Normal.(eta, s)
        end), values = nt -> (; eta = -nt.a)),
    (label = "affine location", data = (; x = X), obs = (; y = Y),
        model = @rkppl(begin
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
        end), values = nt -> (; mu = nt.a .+ nt.b .* X)),
    (label = "chained alias of a location", data = (; x = X), obs = (; y = Y),
        model = @rkppl(begin
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
            mu = a .+ b .* x
            mu2 = mu
            y .~ Normal.(mu2, s)
        end), values = nt -> (; mu = nt.a .+ nt.b .* X, mu2 = nt.a .+ nt.b .* X)),
    (label = "location under a link", data = (; x = X), obs = (; y = COUNTS),
        model = @rkppl(begin
            a ~ Normal(0, 1); b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Poisson.(exp.(eta))
        end), values = nt -> (; eta = nt.a .+ nt.b .* X)),
    (label = "scale predictor", data = (; x = X), obs = (; y = Y),
        model = @rkppl(begin
            a ~ Normal(0, 1); c ~ Normal(0, 1); d ~ Normal(0, 1)
            lsd = c .+ d .* x
            y .~ Normal.(a, exp.(lsd))
        end), values = nt -> (; lsd = nt.c .+ nt.d .* X)),
    (label = "factor location", data = (; g = G), obs = (; y = Y),
        model = @rkppl(begin
            c[levels(g)] .~ Normal.(0, 1); s ~ Exponential(1)
            mu = c[g]
            y .~ Normal.(mu, s)
        end), values = nt -> (; mu = nt.c[G])),
    (label = "matrix location", data = (; x = X), obs = (; y = Y),
        model = @rkppl(begin
            Xm = hcat(x, x .^ 2)
            b[axes(Xm, 2)] .~ Normal.(0, 1); s ~ Exponential(1)
            mu = Xm * b
            y .~ Normal.(mu, s)
        end), values = nt -> (; mu = hcat(X, X .^ 2) * nt.b)),
    (label = "composed location and its sub-predictor", data = (; x = X), obs = (; y = Y),
        model = @rkppl(begin
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
            eta = a .+ b .* x
            mu = exp.(eta)
            y .~ Normal.(mu, s)
        end), values = nt -> (; eta = nt.a .+ nt.b .* X, mu = exp.(nt.a .+ nt.b .* X))),
    (label = "alias of a plate latent", data = (;), obs = (; y = Y),
        model = @rkppl(begin
            s ~ Exponential(1)
            @plate for i in eachindex(y)
                th[i] ~ Normal(0, 1)
            end
            w = th
            y .~ Normal.(w, s)
        end), values = nt -> (; w = nt.th)),
    (label = "alias of a scan state", data = (;), obs = (; y = Y),
        model = @rkppl(begin
            s ~ Exponential(1)
            @scan begin
                u[1] ~ Normal(0, 1)
                for t in 2:T
                    e ~ Normal(0, 1)
                    u[t] = u[t - 1] + e
                end
            end
            w = u
            y .~ Normal.(w, s)
        end), values = nt -> (; w = cumsum(nt._ppl_scan_z_u))),
]

@testset "authored names of optimized definitions are graph values" begin
    for case in CASES
        @testset "$(case.label)" begin
            bound = case.model(; case.data...) | case.obs
            built = build_kernel(bound)
            u = collect(range(-0.4, 0.5; length = built.layout.total))
            nt = constrain(built.layout, u)
            expected = case.values(nt)
            for name in keys(expected)
                @test haskey(built.spec.ports, name)
                value = query(built, bound, name, u)
                # A row-less value reads the same on every location row.
                @test value isa Number || length(value) == length(case.obs.y)
                @test all(isapprox.(value, expected[name]; rtol = 1e-12))
            end
            # No predictor node keeps a generated name in place of its own.
            @test !any(n -> startswith(string(n), "_ppl_lp_"), keys(built.spec.ports))
            # An alias is the node it names: no recipe, coordinate or
            # density of its own.
            plain = ReactiveKernelsPPL._with(bound; named_values = Pair{Symbol,Symbol}[])
            plain_built = build_kernel(plain)
            @test recipes(built.spec) == recipes(plain_built.spec)
            @test coordinate_names(built.layout) == coordinate_names(plain_built.layout)
            @test density(built, bound, u) == density(plain_built, plain, u)
        end
    end
end

@testset "authored names beside several observation axes" begin
    # Two responses with their own rows: each location keeps its name and
    # its own response's rows.
    m = @rkppl begin
        b0 ~ Normal(0, 1); s ~ Exponential(1)
        mu1 = b0 .* x1
        eta2 = b0 .* x2
        y1 .~ Normal.(mu1, s)
        y2 .~ Poisson.(exp.(eta2))
    end
    x1 = [0.1, -0.3, 0.5, 1.2]; x2 = [0.4, -0.2, 0.9]
    bound = m(; x1, x2) | (; y1 = Y, y2 = [1, 0, 2])
    built = build_kernel(bound)
    u = [0.3, 0.1]
    @test query(built, bound, :mu1, u) ≈ 0.3 .* x1
    @test query(built, bound, :eta2, u) ≈ 0.3 .* x2
end

# An inline location has no authored name; its predictor takes the name the
# lowering synthesizes for it (`y_eta`), or the first free `y_eta_k` when the
# model already uses that name, never shadowing it.
const INLINE = @rkppl begin
    a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
    y .~ Normal.(a .+ b .* x, s)
end
const NAMED = @rkppl begin
    a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
end

@testset "an inline location's predictor is named by the lowering" begin
    bound = INLINE(; x = X) | (; y = Y)
    built = build_kernel(bound)
    u = [0.3, -0.2, 0.1]
    @test isempty(bound.named_values)
    @test only(p.name for p in bound.predictors) === :y_eta
    @test query(built, bound, :y_eta, u) ≈ 0.3 .- 0.2 .* X
    # The same predictor under its authored name has the same density.
    nbound = NAMED(; x = X) | (; y = Y)
    @test only(p.name for p in nbound.predictors) === :mu
    @test density(built, bound, u) == density(build_kernel(nbound), nbound, u)
end

@testset "a synthesized predictor name never shadows the model's names" begin
    u = [0.3, -0.2, 0.1]
    expected = density(build_kernel(INLINE(; x = X) | (; y = Y)),
        INLINE(; x = X) | (; y = Y), u)
    # A data column already called `y_eta`.
    data = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
        y .~ Normal.(a .+ b .* x, s)
    end
    dbound = data(; x = X, y_eta = Y .^ 2) | (; y = Y)
    dbuilt = build_kernel(dbound)
    @test only(p.name for p in dbound.predictors) === :y_eta_1
    @test query(dbuilt, dbound, :y_eta_1, u) ≈ 0.3 .- 0.2 .* X
    @test density(dbuilt, dbound, u) == expected
    # A definition already called `y_eta`.
    def = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
        y_eta = exp(a)
        y .~ Normal.(a .+ b .* x, s)
    end
    fbound = def(; x = X) | (; y = Y)
    fbuilt = build_kernel(fbound)
    @test only(p.name for p in fbound.predictors) === :y_eta_1
    @test query(fbuilt, fbound, :y_eta, u) ≈ exp(0.3)
    @test query(fbuilt, fbound, :y_eta_1, u) ≈ 0.3 .- 0.2 .* X
    @test density(fbuilt, fbound, u) == expected
end

@testset "a named value names a defined node" begin
    m = @rkppl begin
        s ~ Exponential(1)
        @plate for i in eachindex(y)
            th[i] ~ Normal(0, 1)
        end
        w = th
        y .~ Normal.(w, s)
    end
    bound = m() | (; y = Y)
    @test bound.named_values == [:w => :th]
    layout = assign_layout(bound)
    # refused: an authored name must read a node of the generated program,
    # never an undefined one (a lowering inconsistency, not a model shape)
    dangling = ReactiveKernelsPPL._with(bound; named_values = [:w => :nowhere])
    @test_throws ContractValidationError kernel_expr(dangling, layout)
    # refused: an authored name cannot shadow another graph value
    shadow = ReactiveKernelsPPL._with(bound; named_values = [:s => :th])
    @test_throws ContractValidationError kernel_expr(shadow, layout)
end

end # module AuthoredNameTests
