# Compiled (Reactant) checks for the cases in test_occupancy.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _occ_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_occ_reactant_measure, built, bound, post_q, u)
end

function _occ_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

# Upstream XLA gap (the besselix/gamma_inc/beta_inc pin precedent):
# broadcast `ifelse.` with an untraced data condition has no Reactant
# materialization rule (`similar` over `Broadcasted{typeof(ifelse)}`
# with a plain `Vector{Bool}` condition), so the masked program fails
# at trace time (measured on Reactant 0.2.288). The signature below is
# exactly that gap; anything else rethrows loudly.
_occ_is_upstream_gap(e) =
    e isa MethodError && e.f === Base.similar && !isempty(e.args) &&
    e.args[1] isa Base.Broadcast.Broadcasted && e.args[1].f === ifelse

@testset "occupancy under Reactant" begin
    icols = _occ_ifelse_cols()
    try
        fx = _occ_reactant(_OCC_IFELSE_PROG, icols)
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.rval ≈ fx.val rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
        # Self-firing pin: errors (Unexpected Pass) once upstream
        # wires the mixed-ifelse broadcast rule, forcing removal of
        # the try/catch.
        @test_broken true
    catch e
        _occ_is_upstream_gap(e) || rethrow()
        # Known upstream mixed-ifelse gap (above): pinned, not passing.
    end
    # Host-masked exact equivalent: full XLA parity on identical math.
    # test_occupancy.jl checks native equality with the ifelse program, so
    # the XLA leg provably covers the same density.
    fx = _occ_reactant(_OCC_MASK_PROG, _occ_masked(icols))
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.rval ≈ fx.val rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
    # Data-length invariance (constraints.md): more rows must not
    # replicate the loop body.
    sdet, ldet = [1, 0], [1, 0, 1, 1, 0, 1]
    small = _occ_reactant(_OCC_MASK_PROG, Dict{Symbol,AbstractVector}(:y => [1, 0],
        :x => [0.0, 1.0], :det => sdet, :mask => _occ_mask(sdet)))
    large = _occ_reactant(_OCC_MASK_PROG, Dict{Symbol,AbstractVector}(
        :y => [1, 0, 1, 0, 1, 0], :x => collect(0.0:5.0), :det => ldet,
        :mask => _occ_mask(ldet)))
    @test small.lines == large.lines
end
