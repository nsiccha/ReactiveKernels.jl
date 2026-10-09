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

# A plate cell whose parameters shadow the child's formals.
@kernel names_cell_child(values, scale) = begin
    pointwise = plate(values, scale) do values, scale
        values / scale
    end
    return pointwise
end

@kernel names_cell_parent(x::Vector{Float64}, s::Float64) = begin
    p = names_cell_child(x, s)
    total = sum(p)
    return total
end

# A typed child formal: an unannotated caller value takes the declared type.
@kernel names_typed_child(v::AbstractVector{Float64}, s) = begin
    total = sum(v) * s
    return total
end

@kernel names_typed_parent(a::Vector{Float64}) = begin
    w = a .+ 1.0
    r = names_typed_child(w, 2.0) + names_typed_child(a .* 2.0, 1.0)
    return r
end

# The kernel's own unannotated input keeps its public contract: the formal
# still converts it.
@kernel names_typed_have(a) = begin
    r = names_typed_child(a, 2.0)
    return r
end

# Two children called inside one expression whose values share names.
@kernel names_rise(x) = begin
    xi = x .* 2.0
    value = xi .+ 1.0
    return value
end

@kernel names_fall(x, xm) = begin
    xi = x .- 3.0
    xi_max = xm .- 3.0
    value = xi .- xi_max
    return value
end

@kernel names_lifted_calls(a::Vector{Float64}, m::Float64) = begin
    mu = a .* names_rise(a) .* exp.(names_fall(a, m))
    twice = names_rise(a .* 3.0) .+ names_rise(m)
    total = sum(mu) + sum(twice)
    return total
end

# A lifted call outside a named assignment keeps a hygienic port; the
# program still reads the child's value by its shown name.
@kernel names_lifted_return(a::Vector{Float64}) = begin
    return sum(names_rise(a) .* 2.0)
end

# An object endpoint evaluated inside a branch arm.
@kernel names_arm(x::Float64, s::Float64) = begin
    lp = s > 0 ? _DKS.normal(0.0, s).logpdf(x) : -Inf
    return lp
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
        # The child's formals read the caller's values in place.
        @test occursin("z = x * y + 1.0", text)
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
        # the two `exponential(4.0)` calls stay distinguishable without ids,
        # and literal arguments read as values.
        @test occursin("var\"lp_s.log_scale\" = log(4.0)", text)
        @test occursin("var\"lp_t.log_scale\" = log(4.0)", text)
        @test occursin("lp_s = ifelse(s >= 0, -var\"lp_s.log_scale\" - s / 4.0, -Inf)", text)
        # `u[2]` is read twice by the endpoint, so it stays one named value.
        @test occursin("var\"lp_t.x\" = u[2]", text)
        # A once-read computed argument reads in place; a bound child
        # endpoint's scoped port reads its own value, not `var"standard.z"`.
        @test occursin("var\"lp_m.standardized\" = (u[3] - -0.5) / 2.0", text)
        @test occursin("var\"lp_m.standard.logpdf\" = -0.5 * log(2π) - 0.5 * " *
                       "var\"lp_m.standardized\" ^ 2", text)
        @test !occursin("standard.z", text)
        @test !occursin("let ", text)
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

    @testset "a do-block parameter shadows only inside the block" begin
        text = _names_text(names_cell_parent)
        @test occursin("p = plate(x, s) do values, scale", text)
        @test !occursin("let ", text)
        @test prepare(names_cell_parent)([1.0, 2.0, 4.0], 2.0) == 3.5
    end

    @testset "structural views read authored source names" begin
        listing = sprint(show, MIME"text/plain"(), kernel_graph(names_priors))
        @test occursin("(z) -> -0.5 * log(2π) - 0.5 * z ^ 2", listing)
        explained = explain(plan(names_priors))
        @test occursin("(z) -> -0.5 * log(2π) - 0.5 * z ^ 2", explained)
    end

    @testset "a typed child formal reads the caller's value in place" begin
        # The unannotated `w` and the computed argument take the formal's
        # declared type, so neither boundary needs an `identity` recipe.
        @test _names_identity_recipes(names_typed_parent) == 0
        @test ReactiveKernels.valtype(names_typed_parent.ports[:w]) ===
              AbstractVector{Float64}
        text = _names_text(names_typed_parent)
        @test occursin("var\"r.names_typed_child\" = sum(w) * 2.0", text)
        @test occursin("var\"r.names_typed_child_2\" = sum(a .* 2.0) * 1.0", text)
        @test !occursin("identity(", text)
        @test !occursin("##", text)
        kernel = prepare(names_typed_parent)
        a = [0.3, -0.2, 0.7]
        reference(a) = sum(a .+ 1.0) * 2.0 + sum(a .* 2.0)
        @test kernel(a) ≈ reference(a)
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        @test gradient(kernel, backend, a) ≈ gradient(reference, backend, a)
        # A HAVE port is the kernel's public boundary: its declared type
        # (here none) is unchanged, so the formal keeps its conversion.
        @test only(inputs(names_typed_have)) isa ReactiveKernels.Value{Any}
        @test occursin("identity(a)", _names_text(names_typed_have))
        @test prepare(names_typed_have)([1.0, 2.0]) == 6.0
    end

    @testset "calls inside an expression are named under the assignment" begin
        text = _names_text(names_lifted_calls)
        # Each call's values read under `<assigned>.<callee>`; a repeated
        # callee in one assignment counts up.
        @test occursin("var\"mu.names_rise.xi\" = a .* 2.0", text)
        @test occursin("var\"mu.names_rise\" = var\"mu.names_rise.xi\" .+ 1.0", text)
        @test occursin("var\"mu.names_fall.xi_max\" = m .- 3.0", text)
        @test occursin("mu = (a .* var\"mu.names_rise\") .* exp.(var\"mu.names_fall\")", text)
        @test occursin("var\"twice.names_rise.xi\" = (a .* 3.0) .* 2.0", text)
        @test occursin("var\"twice.names_rise_2.xi\" = m .* 2.0", text)
        @test occursin("twice = var\"twice.names_rise\" .+ var\"twice.names_rise_2\"", text)
        @test !occursin("##", text)
        @test !occursin("let ", text)
        @test !occursin(r"\b(xi|value)_\d+\b", text)
        kernel = prepare(names_lifted_calls)
        a, m = [0.3, -0.2, 0.7], 0.5
        reference(a) = sum(a .* (a .* 2.0 .+ 1.0) .* exp.((a .- 3.0) .- (m - 3.0))) +
                       sum((a .* 6.0 .+ 1.0) .+ (m * 2.0 + 1.0))
        @test kernel(a, m) ≈ reference(a)
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        @test gradient(a -> kernel(a, m), backend, a) ≈ gradient(reference, backend, a)
    end

    @testset "a hygienic lifted port reads as the value it holds" begin
        # Julia reports a gensym parameter without its leading `##`; the
        # display still maps the source's spelling to the shown value.
        text = _names_text(names_lifted_return)
        @test occursin("value = xi .+ 1.0", text)
        @test !occursin("endpoint_value", text)
        @test prepare(names_lifted_return)([1.0, 2.0]) == 2 * (3.0 + 5.0)
    end

    @testset "an endpoint inside a branch arm reads by its own names" begin
        text = _names_text(names_arm)
        @test occursin("let log_scale = log(s), standardized = (x - 0.0) / s", text)
        @test !occursin("##", text)
        @test !occursin("->", text)
        kernel = prepare(names_arm)
        @test kernel(0.3, 2.0) ≈ _names_normal(0.3, 0.0, 2.0)
        @test kernel(0.3, -1.0) == -Inf
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        @test gradient(u -> kernel(u[1], u[2]), backend, [0.3, 2.0]) ≈
              gradient(u -> _names_normal(u[1], 0.0, u[2]), backend, [0.3, 2.0])
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
