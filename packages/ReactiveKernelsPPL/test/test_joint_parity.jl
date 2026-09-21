# Durable W4 SB-parity test: the final joint program (all 3 configs) binds,
# lays out, and evaluates like+prior at SB point1 theta. Asserts the
# proven-global const to 1e-9 (any drift fails loudly); tighten to absolute
# 1e-9 parity once the delegated const hunt resolves.
include(joinpath(@__DIR__, "parity", "joint_parity_fixture.jl"))

@testset "joint SB parity (W4, const-pinned)" begin
    for config in ("continuous", "ordinal", "binary")
        cols = _parity_columns(config)
        bound =
            bind_data(final_plan(config), cols;
                dims = Dict(:kernel_nsub_pk_loc => 3))
        @test bound.n_obs == 7
        lay = assign_layout(bound)
        nwant = config == "continuous" ? 101 : 96
        @test length(coordinate_names(lay)) == nwant
        k = build_kernel(bound)
        like_q = prepare_query(k, bound, :likelihood)
        prior_q = prepare_query(k, bound, :prior)
        u = unconstrain(lay, final_theta_nt(config))
        got = like_q(u) + prior_q(u)
        gap = got - _PARITY_SB[config]
        @test abs(gap - _PARITY_GAP[config]) < 1e-9
    end
end
