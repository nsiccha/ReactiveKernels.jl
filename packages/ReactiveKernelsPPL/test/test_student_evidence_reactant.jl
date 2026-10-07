# Compiled (Reactant) checks for the cases in test_student_evidence.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size. StudentT is
# verified green on the default pipeline (robust baseline §7n scope
# note), so no ladder-1 pin here.
function _stev_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_stev_reactant_measure, built, bound, post_q, u)
end

function _stev_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    ops = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    return (; lines = count(==('\n'), hlo),
        whiles = count("stablehlo.while", hlo),
        batches = count("enzyme.batch", hlo), ops, native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

# The Student-t evidence arms route through the owned `rk_beta_inc`
# (transparent math over the Student-t slice), which traces under
# Reactant — `SpecialFunctions.beta_inc` has no traced-scalar method
# (upstream Reactant placeholder, no backing MLIR op).
@testset "student evidence under Reactant" begin
    prog = Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        sigma ~ Exponential(1.0)
        y .~ censored.(StudentT.(3.0, mu, sigma), lo, hi)
    end""")
    fx = _stev_reactant(prog, Dict{Symbol,AbstractVector}(
        :y => [0.0, 4.0, 10.0], :x => [0.0, 1.0, 2.0],
        :lo => fill(0.0, 3), :hi => fill(10.0, 3)))
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.rval ≈ fx.val rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
    # Compare the same mix of all three clamp arms at larger row counts.
    # Omitting the interior arm removes its batch entirely during binding;
    # that tests different branch populations rather than length invariance.
    small = _stev_reactant(prog, Dict{Symbol,AbstractVector}(
        :y => repeat([0.0, 4.0, 10.0], 4), :x => collect(0.0:11.0),
        :lo => fill(0.0, 12), :hi => fill(10.0, 12)))
    large = _stev_reactant(prog, Dict{Symbol,AbstractVector}(
        :y => repeat([0.0, 4.0, 10.0], 8), :x => collect(0.0:23.0),
        :lo => fill(0.0, 24), :hi => fill(10.0, 24)))
    for result in (small, large)
        @test result.primal ≈ result.native rtol=1e-9
        @test result.rval ≈ result.val rtol=1e-9
        @test result.rgrad ≈ result.g rtol=1e-7 atol=1e-9
    end
    @test small.whiles == large.whiles
    @test small.batches == large.batches
    @test small.ops == large.ops
end
