using ReactiveKernels
using Test
using LinearAlgebra

# Regression for the deterministic getter-AST fix (src/stateful.jl _ensure_expr):
# pure-path recipe-result locals are named from the stable recipe index, not a
# fresh gensym, so two structurally identical programs share one RGF/getter TYPE
# instead of forcing a full recompile per construction.

_split(x) = (sum(x), x .* 2)                       # (scalar, vector) multi-output

@testset "deterministic getter AST — same-signature repeated dual/welford share type" begin
    @test typeof(dual_averaging_state(0.1)) === typeof(dual_averaging_state(0.2))
    @test typeof(welford_var(3)) === typeof(welford_var(3))
    @test typeof(dual_averaging_state(0.1f0)) !== typeof(dual_averaging_state(0.1))
    @test typeof(welford_var(3, Float32)) !== typeof(welford_var(3, Float64))
end

# Direct prepare_reactive multi-output on the pure default path: a
# multi-output recipe prepared twice from the same spec yields the SAME program type
# and computes correctly.
_det_spec = @kernel begin
    x::Vector{Float64}
    (lo::Float64, hi::Vector{Float64}) = _split(x)
    return (lo, hi)
end

@testset "deterministic getter AST — direct prepare_reactive multi-output" begin
    p1 = prepare_reactive(_det_spec)
    p2 = prepare_reactive(_det_spec)
    @test typeof(p1) === typeof(p2)                          # shared program type
    state = p1([1.0, 2.0, 3.0])
    @test get!(state, statevalue(p1, _det_spec.lo)) == 6.0
    @test get!(state, statevalue(p1, _det_spec.hi)) == [2.0, 4.0, 6.0]
end


# --- poc exact-SHA gate additions ---------------------------------------------
@testset "reactive getter codegen has no active gensym (deterministic AST)" begin
    src = read(joinpath(pkgdir(ReactiveKernels), "src", "stateful.jl"), String)
    lo = findfirst("function _ensure_expr(", src)
    hi = findnext("\nfunction _getter_ast(", src, last(lo))
    fn = src[first(lo):first(hi)]
    code_lines = filter(l -> !occursin(r"^\s*#", l), split(fn, "\n"))
    @test !any(l -> occursin("gensym(", l), code_lines)   # only comments mention it
end

@testset "mixed pure+in-place NUTS group — same signature shares one type" begin
    _mg_grad!(g, q) = (copyto!(g, q); 0.5 * sum(abs2, q))
    metric = Matrix{Float64}(I, 4, 4)
    a = reactive_nuts_group(_mg_grad!, metric, [1.0, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    b = reactive_nuts_group(_mg_grad!, metric, [5.0, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])
    @test typeof(a) === typeof(b)                          # in-place bundle AST stable too
    @test typeof(reactive_program(a)) === typeof(reactive_program(b))
    @test a.dham ≈ b.dham                                 # both compute (behavior intact)
end
