module _QualifiedSubmodelTests
using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels,
    ReactiveKernelsPPL, Test

module _QSLibrary
module Blocks
using ReactiveKernelsPPL

@rkppl inner(x; scale = 1.0) = begin
    b ~ Normal(0, 1)
    return scale .* b .* x
end
@rkppl outer(x; scale = 1.0) = begin
    nested ~ Blocks.inner(x; scale)
    return nested
end
@rkppl stream(x) = begin
    b ~ Normal(0, 1)
    slot .~ Normal.(b .* x, 0.5)
    return slot
end
@rkppl recursive(x) = begin
    nested ~ Blocks.recursive(x)
    return nested
end
end
end

using ._QSLibrary.Blocks: outer, stream
const _QSModuleAlias = _QSLibrary.Blocks
const _QSOuterAlias = outer
const _QSStreamAlias = stream
const _QSMonoAlias = monotonic
const _QS_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const _QS_PROVIDER_CALLS = Ref(0)
_qs_module_provider() = (_QS_PROVIDER_CALLS[] += 1; _QSLibrary.Blocks)

# A caller binding must not change a nested call's defining module.
const inner = 17

const _QS_HEADS = (:outer, :(_QSLibrary.Blocks.outer),
    :(_QSModuleAlias.outer), :_QSOuterAlias,
    GlobalRef(_QSLibrary.Blocks, :outer))
const _QS_STREAM_HEADS = (:stream, :(_QSLibrary.Blocks.stream), :_QSStreamAlias)

function _qs_ast(head; cell = false, stream = false)
    stream && return Expr(:block, Expr(:call, :~, :y, Expr(:call, head, :x)))
    call = Expr(:call, head, Expr(:parameters, Expr(:kw, :scale, 1.3)), :x)
    if cell
        call.args[end] = :(x[i])
        return quote
            @plate for i in eachindex(y)
                z[i] ~ $call
                y[i] ~ Normal(z[i], 0.5)
            end
        end
    end
    return quote
        z_b ~ Normal(0, 1)
        z ~ $call
        y .~ Normal.(z .+ z.nested.b .+ z_b, 0.5)
    end
end

function _qs_build(ast, data)
    plan = lower_rkppl(ast, data; mod = @__MODULE__, conditioned = (:y,))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(data)))
    built = build_kernel(bound)
    u = [0.2 * cos(i) for i in 1:built.layout.total]
    return bound, built, u
end

function _qs_oracle(kind, nt, data)
    if kind === :stream
        return logpdf(Normal(), nt.y.b) +
            sum(logpdf.(Normal.(nt.y.b .* data.x, 0.5), data.y))
    elseif kind === :cell
        b = nt.z.nested.b
        return sum(logpdf.(Normal(), b)) +
            sum(logpdf.(Normal.(1.3 .* b .* data.x, 0.5), data.y))
    end
    b = nt.z.nested.b
    return logpdf(Normal(), b) + logpdf(Normal(), nt.z_b) +
        sum(logpdf.(Normal.(1.3 .* b .* data.x .+ b .+ nt.z_b, 0.5), data.y))
end

function _qs_fd(f, u; h = 1e-6)
    map(eachindex(u)) do i
        up, dn = copy(u), copy(u)
        up[i] += h
        dn[i] -= h
        (f(up) - f(dn)) / (2h)
    end
end

@testset "qualified submodels: transparent binding, density and gradient" begin
    data = (; x = [-0.5, 0.2, 1.0], y = [0.1, -0.2, 0.3])
    saved = deepcopy(data)
    for kind in (:latent, :cell, :stream)
        heads = kind === :stream ? _QS_STREAM_HEADS : _QS_HEADS
        expanded = Expr[]
        coordinates = Vector{Symbol}[]
        for head in heads
            ast = _qs_ast(head; cell = kind === :cell, stream = kind === :stream)
            expansion, _ = ReactiveKernelsPPL._expand_submodels(
                ast, Set((:x, :y)), @__MODULE__)
            push!(expanded, Base.remove_linenums!(deepcopy(expansion)))
            bound, built, u = _qs_build(ast, data)
            push!(coordinates, coordinate_names(built.layout))
            reference(v) = _qs_oracle(kind, constrain(built.layout, v), data)
            q = prepare_sampler(built, bound, u; backend = _QS_BACKEND)
            g = similar(u)
            value, _ = sampler_value_and_gradient!(q, g, u)
            @test value ≈ reference(u) rtol = 1e-12
            @test g ≈ _qs_fd(reference, u) rtol = 1e-5 atol = 1e-7
        end
        @test all(==(first(expanded)), expanded)
        @test all(==(first(coordinates)), coordinates)
    end
    @test data == saved

    # The shipped monotonic submodel expands to its ordinary Julia body.
    data = (; c = [1, 2, 3], y = [0.1, 0.3, 0.8])
    values = Float64[]
    for head in (:monotonic, :(ReactiveKernelsPPL.monotonic), :_QSMonoAlias)
        call = Expr(:call, head, :c, :phi)
        ast = quote
            phi ~ Dirichlet([1.0, 2.0])
            m ~ $call
            y .~ Normal.(m, 0.5)
        end
        bound, built, _ = _qs_build(ast, data)
        u = [0.3]
        reference(v) = begin
            phi = constrain(built.layout, v).phi
            m = cumsum(vcat(0.0, phi))[data.c]
            logpdf(Dirichlet([1.0, 2.0]), phi) + sum(log.(phi)) +
                sum(logpdf.(Normal.(m, 0.5), data.y))
        end
        q = prepare_sampler(built, bound, u; backend = _QS_BACKEND)
        g = similar(u)
        value, _ = sampler_value_and_gradient!(q, g, u)
        push!(values, value)
        @test value ≈ reference(u) rtol = 1e-12
        @test g ≈ _qs_fd(reference, u) rtol = 1e-5 atol = 1e-7
    end
    @test all(≈(first(values)), values)
end

@testset "qualified submodels: binding reads and expansion guards" begin
    # Recursion has no finite transparent expansion (P3).
    err = try
        lower_rkppl(quote
            z ~ _QSLibrary.Blocks.recursive(x)
            y .~ Normal.(z, 1.0)
        end, (:x, :y); mod = @__MODULE__, conditioned = (:y,))
    catch e
        e
    end
    @test err isa SurfaceLoweringError
    @test occursin("calls itself", err.message)

    # A module path is a binding read, never an evaluated expression (P3).
    @test _qs_module_provider() === _QSLibrary.Blocks
    before = _QS_PROVIDER_CALLS[]
    @test_throws SurfaceLoweringError lower_rkppl(quote
        z ~ _qs_module_provider().outer(x)
        y .~ Normal.(z, 1.0)
    end, (:x, :y); mod = @__MODULE__, conditioned = (:y,))
    @test _QS_PROVIDER_CALLS[] == before

    # A qualified ordinary callable is not classified as a submodel.
    @test ReactiveKernelsPPL._resolve_submodel(:(Base.sqrt(x)), @__MODULE__) === nothing
end
end # module _QualifiedSubmodelTests
