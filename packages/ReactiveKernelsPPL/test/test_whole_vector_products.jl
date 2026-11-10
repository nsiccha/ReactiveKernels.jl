module WholeVectorProductTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using Distributions: Normal, logpdf
import DifferentiationInterface as DI
import Enzyme

# A per-observation matrix the graph computes (`(c .* r) .* U`, an N × Q
# matrix: an N-vector broadcast against a data matrix) times a vector of
# data (`w`, Q quadrature weights) is Julia's matrix-vector product: one
# value per observation (snag `rkppl-vector-dat-85351195`). Lowering read
# `w` as an observation column and refused `D * w` as a vector product;
# where `w` was a whole value (`normalized(w)`, or `U * w` over data only)
# it rewrote the product to the elementwise `D .* w`, which fails to
# broadcast when N ≠ Q and silently computes another density when N == Q.

normalized(w) = w ./ sum(w)

const BACKEND = DI.AutoEnzyme(; mode = Enzyme.Reverse)

function fixture(N, Q)
    t = [0.7 * sin(1.3i + 0.2) for i in 1:N]
    c = [0.5 + 0.4 * cos(0.9i) for i in 1:N]
    nodes = [0.1 + 0.8 * (k - 0.5) / Q for k in 1:Q]
    w = [0.2 + 0.1 * sin(2.1k) for k in 1:Q]
    y = [0.6 * cos(1.7i - 0.3) for i in 1:N]
    return (; t, c, nodes, w, y, U = exp.(t .* permutedims(nodes)))
end

prior(u) = logpdf(Normal(), u[1]) + logpdf(Normal(), u[2])
likelihood(f, y) = sum(logpdf.(Normal.(f, 1), y))

# The sampler's value against an independent log density of `(a, b)`, and
# its native Enzyme Reverse gradient against central differences of that
# oracle, at several points. Returns the generated program's text.
function check(model, data, y, oracle)
    bound = model(; data...) | (; y)
    built = build_kernel(bound)
    @test coordinate_names(built.layout) == [:a, :b]
    sampler = prepare_sampler(built, bound, [0.0, 0.0]; backend = BACKEND)
    for u in ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.25])
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(sampler, gradient, u)
        @test value ≈ oracle(u) rtol = 1e-12
        for i in eachindex(u)
            h = 1e-6
            plus, minus = copy(u), copy(u)
            plus[i] += h
            minus[i] -= h
            @test gradient[i] ≈ (oracle(plus) - oracle(minus)) / (2h) rtol = 1e-6 atol = 1e-7
        end
    end
    return string(Base.remove_linenums!(deepcopy(kernel_expr(bound, built.layout))))
end

const SPELLINGS = (
    "data-only matrix definition" => (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1)
        r = exp.(a .+ b .* t)
        U = exp.(t .* permutedims(nodes))
        f = (1 ./ (1 .+ (c .* r) .* U)) * w
        y .~ Normal.(f, 1)
    end), (:t, :c, :nodes, :w)),
    "bound data matrix" => (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1)
        r = exp.(a .+ b .* t)
        f = (1 ./ (1 .+ (c .* r) .* U)) * w
        y .~ Normal.(f, 1)
    end), (:t, :c, :U, :w)),
    "named matrix" => (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1)
        r = exp.(a .+ b .* t)
        D = 1 ./ (1 .+ (c .* r) .* U)
        f = D * w
        y .~ Normal.(f, 1)
    end), (:t, :c, :U, :w)),
    "inline location" => (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1)
        y .~ Normal.((1 ./ (1 .+ (c .* exp.(a .+ b .* t)) .* U)) * w, 1)
    end), (:t, :c, :U, :w)),
    "weights a model function computes" => (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1)
        r = exp.(a .+ b .* t)
        wn = normalized(w)
        f = (1 ./ (1 .+ (c .* r) .* U)) * wn
        y .~ Normal.(f .* sum(w), 1)
    end), (:t, :c, :U, :w)),
    "weights also read whole" => (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1)
        r = exp.(a .+ b .* t)
        sw = sum(w)
        f = (1 ./ (1 .+ (c .* r) .* U)) * (w ./ sw)
        y .~ Normal.(f .* sw, 1)
    end), (:t, :c, :U, :w)),
)

@testset "a computed per-observation matrix times a data vector" begin
    for (N, Q) in ((7, 5), (5, 5), (1, 3))
        fx = fixture(N, Q)
        integrand(u) = 1 ./ (1 .+ (fx.c .* exp.(u[1] .+ u[2] .* fx.t)) .* fx.U)
        oracle(u) = prior(u) + likelihood(integrand(u) * fx.w, fx.y)
        for (name, (model, inputs)) in SPELLINGS
            @testset "N=$N, Q=$Q: $name" begin
                data = NamedTuple{inputs}(map(k -> getfield(fx, k), inputs))
                code = check(model, data, fx.y, oracle)
                # Julia's product, never the elementwise `.*`.
                @test occursin(r"(\)|D) \* \(?(w|wn)\b", code)
                @test !occursin(r"\.\* *\(?(w|wn)\b", code)
            end
        end
    end
end

@testset "a data matrix times a data vector, data only" begin
    for (N, Q) in ((7, 5), (5, 5))
        fx = fixture(N, Q)
        g = fx.U * fx.w
        oracle(u) = prior(u) + likelihood(u[1] .+ u[2] .* g, fx.y)
        for (name, model) in (
            "data" => @rkppl(begin
                a ~ Normal(0, 1); b ~ Normal(0, 1)
                g = U * w
                y .~ Normal.(a .+ b .* g, 1)
            end),
            "a model function's weights" => @rkppl(begin
                a ~ Normal(0, 1); b ~ Normal(0, 1)
                wn = normalized(w)
                g = U * wn
                y .~ Normal.(a .+ b .* (g .* sum(w)), 1)
            end),
        )
            @testset "N=$N, Q=$Q: $name" begin
                check(model, (; fx.U, fx.w), fx.y, oracle)
            end
        end
    end
end

@testset "a vector times a vector stays Julia's refused product" begin
    fx = fixture(7, 7)
    m = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1)
        r = exp.(a .+ b .* t)
        f = r * c
        y .~ Normal.(f, 1)
    end
    err = try
        build_kernel(m(; fx.t, fx.c) | (; fx.y))
        nothing
    catch e
        e
    end
    @test err isa Union{ContractValidationError,ReactiveKernelsPPL.SurfaceLoweringError}
    @test occursin("not elementwise", sprint(showerror, err))
end

end # module
