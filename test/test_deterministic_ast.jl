using ReactiveKernels
using Test
using LinearAlgebra

# Regression for the deterministic getter-AST fix (src/stateful.jl _ensure_expr):
# pure-path recipe-result locals are named from the stable recipe index, not a
# fresh gensym, so two structurally identical programs share one RGF/getter TYPE
# instead of forcing a full recompile per construction.

# A specialize=true object with BOTH a single-output derived node and a MULTI-output
# recipe (tuple assignment), so the multi-output naming path is exercised.
_split(x) = (sum(x), x .* 2)                       # (scalar, vector) multi-output
@reactive specialize=true _det_ast_obj(x) = begin
    total::eltype(x) = sum(x)                      # single-output derived
    (s::eltype(x), doubled::typeof(x)) = _split(x) # multi-output recipe
end

@testset "deterministic getter AST — same signature shares one type" begin
    a = _det_ast_obj([1.0, 2.0, 3.0])
    b = _det_ast_obj([4.0, 5.0, 6.0])
    # Same signature => IDENTICAL object/program/getter TYPES (the whole point).
    @test typeof(a) === typeof(b)
    @test typeof(reactive_program(a)) === typeof(reactive_program(b))
    # And the emitted getter ASTs are identical (no gensym drift).
    prog_a = reactive_program(a); prog_b = reactive_program(b)
    for name in (:total, :s, :doubled)
        ha = getproperty(a.handles, name); hb = getproperty(b.handles, name)
        @test string(code_expr(prog_a, ha)) == string(code_expr(prog_b, hb))
    end
    # Multi-output correctness (behavior regression on the pure default path).
    @test a.total == 6.0
    @test a.s == 6.0
    @test a.doubled == [2.0, 4.0, 6.0]
    @test b.doubled == [8.0, 10.0, 12.0]
end

@testset "deterministic getter AST — distinct signatures stay distinct/concrete" begin
    f64 = _det_ast_obj([1.0, 2.0])
    f32 = _det_ast_obj(Float32[1.0, 2.0])
    # Different storage/precision => distinct concrete types (no over-merge).
    @test typeof(f64) !== typeof(f32)
    @test f32.total isa Float32
    @test f32.doubled isa Vector{Float32}
    @test isconcretetype(typeof(reactive_program(f64)))
    @test isconcretetype(typeof(reactive_program(f32)))
end

@testset "deterministic getter AST — same-signature repeated dual/welford share type" begin
    @test typeof(dual_averaging_state(0.1)) === typeof(dual_averaging_state(0.2))
    @test typeof(welford_var(3)) === typeof(welford_var(3))
    @test typeof(dual_averaging_state(0.1f0)) !== typeof(dual_averaging_state(0.1))
    @test typeof(welford_var(3, Float32)) !== typeof(welford_var(3, Float64))
end

# Direct prepare_reactive multi-output on the pure default path (no facade): a
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

# Diamond: two derived nodes share a common input recipe (re-emitted in both getters).
_shared_add(x) = x .+ 1.0
@reactive specialize=true _diamond_obj(x) = begin
    base::typeof(x) = _shared_add(x)
    left::eltype(x) = sum(base)
    right::eltype(x) = sum(base .* base)
end

@testset "diamond / shared-input re-emission — same type, correct values" begin
    a = _diamond_obj([1.0, 2.0, 3.0])
    b = _diamond_obj([4.0, 5.0, 6.0])
    @test typeof(a) === typeof(b)
    @test a.left == 9.0                                   # sum([2,3,4])
    @test a.right == 4.0 + 9.0 + 16.0                     # sum([2,3,4].^2)
    @test b.left == sum([5.0, 6.0, 7.0])
end

# Two structurally DIFFERENT definitions with the SAME recipe count must stay
# distinct getter/program types (deterministic names must not collapse them).
@reactive specialize=true _struct_a(x) = begin
    p::eltype(x) = sum(x)
    q::eltype(x) = maximum(x)
end
@reactive specialize=true _struct_b(x) = begin
    p::eltype(x) = prod(x)
    q::eltype(x) = minimum(x)
end

@testset "distinct structure, same recipe count — no wrong cache collapse" begin
    a = _struct_a([2.0, 3.0, 4.0])
    b = _struct_b([2.0, 3.0, 4.0])
    @test typeof(a) !== typeof(b)
    @test typeof(reactive_program(a)) !== typeof(reactive_program(b))
    @test (a.p, a.q) == (9.0, 4.0)                        # sum, maximum
    @test (b.p, b.q) == (24.0, 2.0)                       # prod, minimum
end
