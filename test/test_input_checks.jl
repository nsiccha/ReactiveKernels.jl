module InputCheckTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
const RK = ReactiveKernels

# A packed vector whose blocks start where data says: `a` is `x[1:n]` and `b`
# the coordinate after it, as ReactiveKernelsPPL reads a binding's positions.
# Without a check, a longer `x` is read at these positions and a shorter one
# fails at whichever read passes its end (snag `rk-prepared-cut-0e0b98ed`).
input_check_length(x, total) = (length(x) == total ||
    throw(DimensionMismatch("packed length $(length(x)) ≠ total $total")); nothing)

@kernel _input_check_blocks(x::AbstractVector{Float64}, n::Int) = begin
    stop::Int = n + 1
    a = view(x, 1:n)
    b = x[stop]
    total = sum(a) + 10 * b
end
RK.input_check!(_input_check_blocks, :x, input_check_length, :x, :stop)

@testset "an input check runs at every cut's entry" begin
    spec = _input_check_blocks
    x3, x4 = [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]
    refused = "packed length"
    # Bound data: the total is computed when the data is bound.
    cut = prepare(spec; have = (:x, :n), want = :total, bound = (; n = 2))
    @test cut(x3) == 33.0
    @test_throws refused cut(x4)
    @test_throws refused cut(x3[1:2])
    @test occursin("_input_check(", string(code_expr(cut)))
    @test occursin("input_check_length", string(code_expr(cut)))
    # Live data: the cut computes the total, which it does not return.
    live = prepare(spec; have = (:x, :n), want = :total)
    @test live(x3, 2) == 33.0
    @test_throws refused live(x4, 2)
    @test live(x4, 3) == 46.0
    # A cut that does not read the input runs no check.
    size_only = prepare(spec; have = (:x, :n), want = :stop)
    @test size_only(x4, 2) == 3
    @test !occursin("input_check_length", string(code_expr(size_only)))
    # Each lane of a vectorized cut.
    lanes = vectorize(cut; batched = :x)
    @test lanes(hcat(x3, 2x3)) == [33.0, 66.0]
    @test_throws refused lanes(hcat(x4, x4))
    # The primal of a derivative operator. A non-allocating kernel's check is
    # tested with MutatingFunctions, in `test_nonallocating.jl`.
    ad = prepare_ad(cut, AutoEnzyme(; mode = Enzyme.Reverse), x3; active = :x)
    value, gradient = ad_value_and_gradient(ad, x3)
    @test value == 33.0 && gradient == [1.0, 1.0, 10.0]
    @test_throws refused ad_value_and_gradient(ad, x4)
end

# Enzyme never inlines `_input_check`, which it treats as constant, so a
# gradient calls it as a compiled method. Each compiled method is specialized
# on the check's arguments. An unspecialized `Vararg` method was a generic call
# through the boxed calling convention, 10 to 14 ns on every gradient of a
# small model (snag `packed-point-inp-ce5df38b`).
@testset "a gradient calls the input check specialized" begin
    x3 = [1.0, 2.0, 3.0]
    cut = prepare(_input_check_blocks; have = (:x, :n), want = :total, bound = (; n = 2))
    ad = prepare_ad(cut, AutoEnzyme(; mode = Enzyme.Reverse), x3; active = :x)
    @test ad_value_and_gradient(ad, x3) == (33.0, [1.0, 1.0, 10.0])
    compiled = [Base.unwrap_unionall(mi.specTypes)
        for m in methods(RK._input_check) for mi in Base.specializations(m) if mi !== nothing]
    @test !isempty(compiled)
    @test !any(T -> Base.isvarargtype(last(T.parameters)), compiled)
end

# Kernels derived from a checked one keep its check on the same value.
@kernel _input_check_summary(total) = begin
    doubled = 2 * total
end
@kernel _input_check_caller(v::AbstractVector{Float64}, m::Int) = begin
    t = _input_check_blocks(v, m)
    halved = t / 2
end

@testset "derived kernels keep the check" begin
    x3, x4 = [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]
    refused = "packed length"
    composed = compose(_input_check_blocks, _input_check_summary)
    cut = prepare(composed; have = (:x, :n), want = :doubled, bound = (; n = 2))
    @test cut(x3) == 66.0
    @test_throws refused cut(x4)
    renamed = RK.rename(_input_check_blocks, :x => :y)
    cut = prepare(renamed; have = (:y, :n), want = :total, bound = (; n = 2))
    @test cut(x3) == 33.0
    @test_throws refused cut(x4)
    # A call of the kernel inside another checks the caller's argument.
    cut = prepare(_input_check_caller; have = (:v, :m), want = :halved, bound = (; m = 2))
    @test cut(x3) == 16.5
    @test_throws refused cut(x4)
    # The checked kernel itself is unchanged.
    @test length(_input_check_blocks.graph.checks) == 1
end

@kernel _input_check_names(x::Vector{Float64}) = begin
    y = 2 .* x
end
@testset "input_check! names values and refuses what it cannot run" begin
    @test_throws "not a value of this kernel" RK.input_check!(_input_check_names,
        :z, input_check_length, :z, 1)
    @test_throws "bound to a global name" RK.input_check!(_input_check_names,
        :x, (x, n) -> nothing, :x, 1)
end
end
