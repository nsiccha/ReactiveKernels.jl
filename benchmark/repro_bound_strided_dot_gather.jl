# Shared-core companion to repro_reactant_strided_dot_gather.jl.
# Bound Boolean branches partition a batched plate into strided row gathers.
# Native values/ordinary reverse and the full zero-filled output are the oracle.
# No PPL response policy, compiler default or derivative rule is changed here.
#
# julia --project=<RK/Reactant/Enzyme environment> benchmark/repro_bound_strided_dot_gather.jl
# ... omit_slice_dot_general  # diagnostic bisection only
using ReactiveKernels, Reactant, Enzyme, DifferentiationInterface, Test

Reactant.set_default_backend("cpu")
@kernel function bound_strided_dot(u::Vector{Float64}, design::Matrix{Float64},
                                  present::Vector{Bool})
    mu = design * u
    pointwise = plate(mu, present) do m, has
        value::Float64 = has ? m * m : 0.0
        value
    end
    total = sum(pointwise)
end

mode = isempty(ARGS) ? "default" : only(ARGS)
mode in ("default", "omit_slice_dot_general") ||
    error("expected default or omit_slice_dot_general")
optimize = if mode == "default"
    true
else
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
    pipeline = ext._rk_reactant_default_pipeline("cpu")
    pattern = r"slice_dot_general<\d+>;"
    occursin(pattern, pipeline) || error("slice_dot_general pattern is absent")
    replace(pipeline, pattern => "")
end

for n in (5, 9)
    design = hcat(ones(n), collect(range(-0.4, 0.6; length=n)))
    present = [isodd(i) for i in 1:n]
    saved = deepcopy((design, present))
    total = prepare(bound_strided_dot; want=:total, bound=(; design, present))
    pointwise = prepare(bound_strided_dot; want=:pointwise, bound=(; design, present))
    u = [0.2, -0.3]
    expected = [present[i] ? (design[i, 1]*u[1] + design[i, 2]*u[2])^2 : 0.0
                for i in 1:n]
    expected_gradient = [sum(2*(design[i, 1]*u[1] + design[i, 2]*u[2])*design[i, j]
                             for i in 1:n if present[i]) for j in 1:2]
    @test total(u) ≈ sum(expected)
    @test pointwise(u) ≈ expected
    ad = prepare_ad(bound_strided_dot, AutoEnzyme(; mode=Enzyme.Reverse), u;
                    active=:u, want=:total, bound=(; design, present))
    _, native_gradient = ad_value_and_gradient(ad, u)
    @test native_gradient ≈ expected_gradient
    ru = Reactant.to_rarray(u)
    raw = repr(Reactant.@code_hlo optimize=false total(ru))
    if haskey(ENV, "RK_STRIDED_DOT_IR_DIR")
        dir = ENV["RK_STRIDED_DOT_IR_DIR"]
        mkpath(dir)
        write(joinpath(dir, "bound-$n-raw.mlir"), raw)
    end
    println("BOUND_STRIDED_DOT_BEGIN n=", n, " mode=", mode)
    flush(stdout)
    compiled = Reactant.compile(total, (ru,); optimize)
    compiled_pointwise = Reactant.compile(pointwise, (ru,); optimize)
    compiled_reverse = compile_ad_value_and_gradient(ad, ru; optimize)
    @test Float64(compiled(ru)) ≈ sum(expected)
    @test Array(compiled_pointwise(ru)) ≈ expected
    @test size(Array(compiled_pointwise(ru))) == (n,)
    value, gradient = compiled_reverse(ru)
    @test Float64(value) ≈ sum(expected)
    @test Array(gradient) ≈ expected_gradient
    @test Array(ru) == u
    @test isequal((design, present), saved)
    println("BOUND_STRIDED_DOT_PASS n=", n, " mode=", mode)
end
