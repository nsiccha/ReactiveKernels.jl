using ReactiveKernels
using ReactiveKernelsPPL
using MutatingFunctions
using DifferentiationInterface: AutoEnzyme
import Enzyme
using Test

# Storage reuse and the opt-in generated reverse must compose. A matching
# gradient alone would miss the original bug: the explicit nonallocating
# request silently omitted every generated rule.
module NonallocPlateReverseModels
using ReactiveKernelsPPL
const grouped = @rkppl begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    z[1:J] .~ Normal.(0, 1)
    mu = a .+ z[group]
    y .~ Normal.(mu, sigma)
end

const regression = @rkppl begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    beta[axes(X, 2)] .~ Normal.(0, 1)
    mu = a .+ X * beta
    y .~ Normal.(mu, sigma)
end
end

_napr_rule_count(x::ReactiveKernels._PlateSumRule) = 1
_napr_rule_count(x::Expr) = sum(_napr_rule_count, x.args; init = 0)
_napr_rule_count(x) = 0

_napr_structure(x) = (1, 0)
_napr_structure(x::ReactiveKernels._PlateProgram) = _napr_structure(x.ast)
_napr_structure(x::ReactiveKernels._PlateCall) =
    _napr_structure(ReactiveKernels._program_expression(x.program))
function _napr_structure(x::ReactiveKernels._PlateSumRule)
    primal = _napr_structure(x.program)
    reverse = _napr_structure(x.reverse)
    primal .+ reverse
end
function _napr_structure(x::Expr)
    children = map(_napr_structure, x.args)
    (1 + sum(first, children; init = 0),
     (x.head in (:for, :while)) + sum(last, children; init = 0))
end

function _napr_installed(k::NonAllocatingKernel, u, enabled)
    with_plate_reverse(enabled) do
        f, _, _, _, _ = ReactiveKernels._ad_na_program(k, 1, (u,))
        _napr_rule_count(ReactiveKernels._program_expression(f))
    end
end
function _napr_installed(k::PreparedKernel, u, enabled)
    with_plate_reverse(enabled) do
        _napr_rule_count(ReactiveKernels._ad_inline_plate_programs(k, 1).ast)
    end
end

@testset "nonallocating generated plate reverse" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    structures = Dict()
    for n in (0, 7, 73), model in (:grouped, :regression)
        @testset "$model, $n observations" begin
            y = [0.2 * sin(i) for i in 1:n]
            X = [cos(i + j) for i in 1:n, j in 1:2]
            group = [mod1(i, 2) for i in 1:n]
            originals = deepcopy((y, X, group))
            plan = if model === :grouped
                NonallocPlateReverseModels.grouped(; group, J = 2) | (; y)
            else
                NonallocPlateReverseModels.regression(; X) | (; y)
            end
            built = build_kernel(plan)
            u = [0.1 * sin(i) for i in 1:built.layout.total]
            queries = Dict()
            for nonallocating in (false, true, :auto), enabled in (false, true)
                q = with_plate_reverse(enabled) do
                    prepare_sampler(built, plan, u; backend, nonallocating)
                end
                queries[(nonallocating, enabled)] = q
                @test (q.kernel isa NonAllocatingKernel) == (nonallocating !== false)
                installed = _napr_installed(q.kernel, u, enabled)
                @test enabled ? installed > 0 : installed == 0
                report = plate_reverse_report(q.kernel)
                opposite_report = with_plate_reverse(!enabled) do
                    plate_reverse_report(q.kernel)
                end
                @test report == opposite_report
                # The report describes the prepared representation. Native
                # optimizations selected with the rule disabled can produce
                # forms outside the generated reverse's current coverage.
                if enabled
                    @test any(r -> last(r) == "rule", report)
                    if model === :regression
                        @test (:_ppl_lik_y_resp => "rule") in report
                    end
                end
                if n != 0
                    ast = with_plate_reverse(enabled) do
                        q.kernel isa NonAllocatingKernel ?
                            ReactiveKernels._program_expression(first(
                                ReactiveKernels._ad_na_program(q.kernel, 1, (u,)))) :
                            ReactiveKernels._ad_inline_plate_programs(q.kernel, 1).ast
                    end
                    structure = _napr_structure(ast)
                    @test last(structure) > 0
                    key = (model, nonallocating, enabled)
                    if n == 7
                        structures[key] = structure
                    else
                        @test structure == structures[key]
                    end
                end
            end
            for point in (u, u .+ 0.3, u .- 0.2, u)
                original_point = copy(point)
                reference_gradient = similar(point)
                reference, _ = sampler_value_and_gradient!(
                    queries[(false, false)], reference_gradient, point)
                for q in values(queries)
                    gradient = similar(point)
                    value, _ = sampler_value_and_gradient!(q, gradient, point)
                    @test value ≈ reference rtol = 1e-12
                    @test gradient ≈ reference_gradient rtol = 1e-10 atol = 1e-12
                end
                @test point == original_point
            end
            # Reusing one query must not affect another query's caches or the
            # caller's bound operands.
            @test isequal((y, X, group), originals)
            @test isequal(u, [0.1 * sin(i) for i in 1:built.layout.total])
        end
    end
end

@testset "plate axis markers cannot hide numerical alias reads" begin
    RK = ReactiveKernels
    prefix = quote
        local marker
        marker = $(GlobalRef(RK, :_authored_plate_marker))(Val(()), x)
        shape = Base.axes(marker)
        element = Base.eltype(marker)
        broadcastable = Base.broadcastable(x)
        prepared = Base.Broadcast.preprocess(nothing, broadcastable)
        total = $(GlobalRef(RK, :_plate_total_seed))(Float64, 0.0)
    end
    suffix = quote
        out = $(GlobalRef(RK, :_plate_total_value))(total)
        out
    end
    program(extra) = Expr(:function, Expr(:tuple, :x),
        Expr(:block, prefix.args..., extra..., suffix.args...))
    # The marker is an array alias, but shape-only uses need no adjoint.
    reverse, scalar, batched = RK._pr_program_reverse(program(Any[]), [:x], RK,
        Dict{Symbol,Any}())
    @test isempty(scalar)
    @test batched == [:x]
    for extra in (
        Any[:(term = Base.sum(marker)),
            :(total = $(GlobalRef(RK, :_plate_total_add))(total, term))],
        Any[:(alias = marker), :(term = Base.sum(alias)),
            :(total = $(GlobalRef(RK, :_plate_total_add))(total, term))],
        Any[:(marker[1] = x[1] * x[1])],
        Any[:(marker = Base.fill!(x, 0.0))],
    )
        err = try
            RK._pr_program_reverse(program(extra), [:x], RK, Dict{Symbol,Any}())
            nothing
        catch error
            error
        end
        @test err isa RK._PRIneligible
        @test err isa RK._PRIneligible && occursin("also read directly by `marker`", err.reason)
    end
end

@kernel napr_changing_shape(x::Vector{Float64}) = begin
    squared = x .* x
    fourth = plate(squared) do t
        cell::Float64 = t * t
        cell
    end
    scale::Float64 = sum(fourth)
    shifted = squared .+ x
    weighted = plate(shifted, scale) do t, s
        cell::Float64 = s * sin(t)
        cell
    end
    total::Float64 = scale + sum(weighted)
    return total
end

@testset "plate reverse with changed cache shapes" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    exemplar = [0.1 * sin(i) for i in 1:7]
    kernel = prepare(napr_changing_shape; have = (:x,), want = :total)
    prepared = map((false, true)) do enabled
        with_plate_reverse(enabled) do
            nonallocating = prepare_nonallocating(kernel, exemplar)
            nonallocating, prepare_ad(nonallocating, backend, exemplar; active = :x)
        end
    end
    @test _napr_installed(first(prepared[2]), exemplar, true) > 0
    # Retain the late-enable case too: earlier native transformations may
    # require backend fallback, whose values and changed-shape gradients must
    # still agree. This is not evidence of generated-rule engagement.
    late_kernel = first(prepared[1])
    late_ad = with_plate_reverse(true) do
        prepare_ad(late_kernel, backend, exemplar; active = :x)
    end
    for n in (7, 73, 0, 11, 7)
        x = [0.1 * cos(i) for i in 1:n]
        original = copy(x)
        scale = sum(t -> t^4, x; init = 0.0)
        factor = 1 + sum(t -> sin(t^2 + t), x; init = 0.0)
        expected = scale * factor
        derivative = 4 .* x.^3 .* factor .+
            scale .* cos.(x.^2 .+ x) .* (2 .* x .+ 1)
        for (nonallocating, ad) in (prepared..., (late_kernel, late_ad))
            gradient = similar(x)
            value, _ = ad_value_and_gradient!(ad, gradient, x)
            @test value ≈ expected rtol = 1e-12
            @test gradient ≈ derivative rtol = 1e-10 atol = 1e-12
            @test nonallocating(x) ≈ expected rtol = 1e-12
            @test x == original
        end
    end
end
