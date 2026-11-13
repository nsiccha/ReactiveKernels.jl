module PlateReverseSingleTests
using ReactiveKernels, MutatingFunctions, DifferentiationInterface, Enzyme, Test
const RK = ReactiveKernels

@kernel single_plate(x::Vector{Float64}, s::Float64) = begin
    cells = plate(x) do xi
        sin(s * xi)
    end
    total::Float64 = sum(cells)
    return total
end

@kernel single_factored(x::Vector{Float64}, s::Float64) = begin
    cells = plate(x) do xi
        -0.5 * (xi / s)^2 - log(s)
    end
    total::Float64 = sum(cells)
    return total
end

unruled_density(x) = sin(x) + cos(x)
@kernel single_fallback(x::Vector{Float64}, s::Float64) = begin
    cells = plate(x) do xi
        unruled_density(s * xi)
    end
    total::Float64 = sum(cells)
    return total
end

count_rules(x::RK._PlateSumRule) = 1
count_rules(x::Expr) = sum(count_rules, x.args; init = 0)
count_rules(x) = 0
count_calls(x::RK._PlateCall) = 1
count_calls(x::Expr) = sum(count_calls, x.args; init = 0)
count_calls(x) = 0
function programs!(found, ex)
    ex isa RK._PlateProgram && push!(found, ex)
    ex isa Expr && foreach(a -> programs!(found, a), ex.args)
    found
end
ad_ast(k::PreparedKernel, args) = RK._ad_inline_plate_programs(k, (1, 2)).ast
ad_ast(k::NonAllocatingKernel, args) =
    RK._program_expression(first(RK._ad_na_program(k, (1, 2), args)))

@testset "rule fallback retains native direct reads" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    x0, s0 = [0.2, -0.3], 0.7
    for prepare_enabled in (false, true), reuse in (false, true)
        plain = with_plate_reverse(prepare_enabled) do
            prepare(single_fallback)
        end
        kernel = reuse ? prepare_nonallocating(plain, x0, s0) : plain
        @test occursin("_plate_read", string(kernel.ast))
        report = plate_reverse_report(kernel, (1, 2))
        @test length(report) == 1
        @test occursin("unruled_density", last(only(report)))
        ad = with_plate_reverse(true) do
            @test count_rules(ad_ast(kernel, (x0, s0))) == 0
            prepare_ad(kernel, backend, x0, s0; active = (:x, :s))
        end
        for n in (0, 1, 73)
            x = [0.1 * sin(i) for i in 1:n]
            value, (dx, ds) = ad_value_and_gradient(ad, x, s0)
            derivative = cos.(s0 .* x) .- sin.(s0 .* x)
            @test value ≈ sum(unruled_density, s0 .* x; init = 0.0)
            @test dx ≈ s0 .* derivative
            @test ds ≈ sum(x .* derivative; init = 0.0)
        end
    end
end

@testset "one plate retains its AD boundary and ordinary inline execution" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    x0, s0 = [0.3, -0.2], 0.7
    for prepare_enabled in (false, true), reuse in (false, true)
        plain = with_plate_reverse(prepare_enabled) do
            prepare(single_plate)
        end
        programs = programs!(RK._PlateProgram[], plain.ast)
        @test length(programs) == 1
        @test only(programs).always_inline
        # Compatibility for internal producers of ordinary called programs.
        @test !RK._PlateProgram(only(programs).ast, nothing).always_inline
        @test deepcopy(only(programs)) === only(programs)
        native = plain.f isa RK._ArrayFunction ? plain.f.native : plain.f
        @test count_calls(RK._program_expression(native)) == 0
        kernel = reuse ? prepare_nonallocating(plain, x0, s0) : plain
        report = plate_reverse_report(kernel, (1, 2))
        @test length(report) == 1
        @test last(only(report)) == "rule"
        for enabled in (false, true)
            ad = with_plate_reverse(enabled) do
                @test count_rules(ad_ast(kernel, (x0, s0))) == Int(enabled)
                prepare_ad(kernel, backend, x0, s0; active = (:x, :s))
            end
            point, contexts = RK._ad_prepared_arguments(ad, (x0, s0), NamedTuple())
            @test isempty(RK._code_findings(ad.call,
                RK._ad_call_argument_types(point, contexts);
                recurse = RK._code_recursion(Val(:all))))
            for n in (0, 1, 7, 73), s in (0.7, -0.3)
                x = [0.1 * cos(i) for i in 1:n]
                before = copy(x)
                value, (dx, ds) = ad_value_and_gradient(ad, x, s)
                @test value ≈ sum(sin, s .* x; init = 0.0)
                @test dx ≈ s .* cos.(s .* x)
                @test ds ≈ sum(x .* cos.(s .* x); init = 0.0)
                @test x == before
            end
        end
    end
end

@testset "one factored plate reports and preserves its fallback" begin
    kernel = with_plate_reverse(false) do
        prepare(single_factored)
    end
    report = plate_reverse_report(kernel, (1, 2))
    @test length(report) == 1
    @test occursin("intermediate plate total", last(only(report)))
    @test occursin("factored_partial", string(kernel.ast))
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    ad = with_plate_reverse(true) do
        @test count_rules(ad_ast(kernel, ([0.2], 0.7))) == 0
        prepare_ad(kernel, backend, [0.2], 0.7; active = (:x, :s))
    end
    for n in (0, 1, 73)
        x, s = [0.1 * sin(i) for i in 1:n], 0.7
        value, (dx, ds) = ad_value_and_gradient(ad, x, s)
        @test value ≈ sum(@. -0.5 * (x / s)^2 - log(s))
        @test dx ≈ -x ./ s^2
        @test ds ≈ sum(@. x^2 / s^3 - 1 / s)
    end
end
end
