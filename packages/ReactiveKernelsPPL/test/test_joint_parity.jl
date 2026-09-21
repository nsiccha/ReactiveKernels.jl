# Durable W4 SB-parity test: the final joint program (all 3 configs) binds,
# lays out, and evaluates the full posterior at SB point1 theta. Both
# sides share Stan's u-space (vine transplant), so raw SB lp compares
# directly — no constrained-space workaround. Asserts the proven-global
# const to 1e-9 (any drift fails loudly); tighten to absolute 1e-9
# parity once the delegated const hunt resolves.
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
        post_q = prepare_query(k, bound, :sampler)
        u = unconstrain(lay, final_theta_nt(config))
        # Same u-space: RKPPL LKJ segments equal SB's vine coords.
        z7, z2 = _PARITY_LKJ_Z[config]
        for e in lay.entries
            e.kind === :varying_corr || continue
            seg = u[e.offset:(e.offset + e.size - 1)]
            e.name === :L_subject && @test seg ≈ z7
            e.name === :L_subject_d_tg && @test seg ≈ z2
        end
        got = post_q(u)
        gap = got - _PARITY_SB[config]
        @test abs(gap - _PARITY_GAP[config]) < 1e-9
    end
end

# The generated program's statement count is O(1) in the data: grouped
# cell assignments emit one subject-batched call each (the runner loops
# over `op_ends` at runtime), so tiling the fixture must not change the
# emitted program's shape — only the layout and the bound columns grow.
isdefined(@__MODULE__, :tiled_columns) ||
    include(joinpath(@__DIR__, "parity", "joint_tiling.jl"))

function _parity_statement_heads(K)
    bound = bind_data(final_plan("continuous"), tiled_columns(K);
        dims = Dict(:kernel_nsub_pk_loc => 3K))
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        h = string(st.head)
        if h == "=" && st.args[2] isa Expr && st.args[2].head === :call &&
                st.args[2].args[1] isa Symbol
            h = "=call:" * string(st.args[2].args[1])
        end
        heads[h] = get(heads, h, 0) + 1
    end
    return heads, def
end

@testset "grouped emission is O(1) in the subject count" begin
    h1, def1 = _parity_statement_heads(1)
    h3, _ = _parity_statement_heads(3)
    h10, _ = _parity_statement_heads(10)
    @test h1 == h3
    @test h1 == h10
    @test h1["=call:linear_pk_read_locs_auc_over_subjects"] == 1
    @test h1["=call:tgi_segmented_nadir"] == 1
    r = repr(def1)
    @test !occursin("_s1", r)
    @test !occursin("view(pk_sched_", r)
    @test !occursin("view(log_F", r)
    @test !occursin("scan(", r)
end
