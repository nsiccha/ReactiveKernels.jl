using ReactiveKernels
using ReactiveKernelsDistributionKernels: DistributionKernelSources
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
using Test

# A composed child kernel or object endpoint keeps the caller's names: the
# value the caller assigns keeps its name, child formals read the caller's
# values (no `identity` aliases), the child's result is not `__return__`, the
# child's own values are scoped under the caller's name, and arguments RK
# materializes into hygienic ports show as their values in `readable_code`.

const _DKS = DistributionKernelSources

@kernel names_child(u, w) = begin
    return u * w + 1.0
end

@kernel names_parent(x::Float64, y::Float64) = begin
    z::Float64 = names_child(x, y)
    q = 2 * z
    return q
end

@kernel names_pair(n) = begin
    m = n + 1
    k = m * 2
    return m, k
end

@kernel names_pair_caller(n::Int) = begin
    (a::Float64, b) = names_pair(n)
    total = a + b
    return total
end

@kernel names_priors(u::Vector{Float64}) = begin
    s::Float64 = exp(u[1])
    lp_s = _DKS.exponential(4.0).logpdf(s)
    lp_t = _DKS.exponential(4.0).logpdf(u[2])
    lp_m = _DKS.normal(-0.5, 2.0).logpdf(u[3])
    total = lp_s + lp_t + lp_m
    return total
end

@kernel names_lifted(x::Float64, y::Float64) = begin
    total = _DKS.normal(0.0, 1.0).logpdf(x) + _DKS.normal(0.0, 1.0).logpdf(y)
    return total
end

_names_exponential(x, scale) = x >= 0 ? -log(scale) - x / scale : -Inf
_names_normal(x, location, scale) =
    -0.5 * log(2π) - 0.5 * ((x - location) / scale)^2 - log(scale)
_names_priors_reference(u) = _names_exponential(exp(u[1]), 4.0) +
    _names_exponential(u[2], 4.0) + _names_normal(u[3], -0.5, 2.0)

_names_text(spec) = string(readable_code(spec))
_names_identity_recipes(spec) =
    count(recipe -> recipe.op === identity, kernel_graph(spec).recipes)

@testset "composed kernels keep the caller's names" begin
    @testset "unannotated child boundaries read the caller's values" begin
        @test _names_identity_recipes(names_parent) == 0
        text = _names_text(names_parent)
        @test occursin("z = let u = x, w = y", text)
        @test !occursin("__return__", text)
        @test !occursin("identity(", text)
        @test prepare(names_parent)(1.5, 2.0) == 2 * (1.5 * 2.0 + 1.0)
    end

    @testset "a declared result read inside the child keeps its conversion" begin
        # `m` is the declared `a` and is also read by the child's `k`: the
        # conversion stays its own recipe, so `k` reads the unconverted `m`.
        # (Its InverseFunctions inverse edge `m = identity(a)` is synthesized
        # as for any identity recipe.) The unannotated `n` and `b` need none.
        conversions = [recipe for recipe in kernel_graph(names_pair_caller).recipes
                       if recipe.op === identity]
        @test sort!([only(recipe.outputs).name for recipe in conversions]) == [:a, :m]
        @test occursin("a = identity(m)", _names_text(names_pair_caller))
        @test prepare(names_pair_caller)(3) == 4 + 8
    end

    @testset "endpoint calls: caller names, scoped internals, literal values" begin
        text = _names_text(names_priors)
        @test !occursin("##", text)
        @test !occursin("_binding", text)
        @test !occursin("_argument", text)
        @test !occursin("__return__", text)
        @test !occursin(r"\b(logpdf|log_scale|constrain)_\d+\b", text)
        # The child's own values are named under the caller's result name, so
        # the two `exponential(4.0)` calls stay distinguishable without ids.
        @test occursin("var\"lp_s.log_scale\" = let scale = 4.0", text)
        @test occursin("var\"lp_t.log_scale\" = let scale = 4.0", text)
        @test occursin("lp_s = let x = s,", text)
        @test occursin("lp_t = let x = u[2],", text)
        # A bound child endpoint's scoped port binds its source's own name.
        @test occursin("var\"lp_m.standard.logpdf\" = let z = var\"lp_m.standardized\"", text)
        @test !occursin("let var\"standard.z\"", text)
        parsed = Meta.parseall(text)
        @test !any(arg -> arg isa Expr && arg.head === :error, parsed.args)

        kernel = prepare(names_priors)
        u = [0.3, 0.7, -1.1]
        @test kernel(u) ≈ _names_priors_reference(u)
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        @test gradient(kernel, backend, u) ≈
              gradient(_names_priors_reference, backend, u)
    end

    @testset "a lifted endpoint subexpression reads as authored" begin
        text = _names_text(names_lifted)
        @test !occursin("##", text)
        @test prepare(names_lifted)(0.4, -1.2) ≈
              _names_normal(0.4, 0.0, 1.0) + _names_normal(-1.2, 0.0, 1.0)
    end

    @testset "structural views read authored source names" begin
        listing = sprint(show, MIME"text/plain"(), kernel_graph(names_priors))
        @test occursin("(z) -> -0.5 * log(2π) - 0.5 * z ^ 2", listing)
        explained = explain(plan(names_priors))
        @test occursin("(z) -> -0.5 * log(2π) - 0.5 * z ^ 2", explained)
    end

    @testset "merging a spec with unannotated aliases" begin
        fragment = @kernel begin
            q
            r = q + 1
        end
        merged = merge(names_parent, fragment)
        @test prepare(merged; have = (:x, :y), want = :r)(1.5, 2.0) ==
              2 * (1.5 * 2.0 + 1.0) + 1
    end
end
