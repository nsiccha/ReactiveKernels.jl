using Reactant

function _prv_operations(hlo)
    operations = String[]
    function walk(op)
        push!(operations, Reactant.MLIR.IR.name(op))
        for region in op, block in region, child in block
            walk(child)
        end
    end
    Reactant.MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        mod = parse(Reactant.MLIR.IR.Module, String(hlo); context = ctx)
        try
            walk(Reactant.MLIR.IR.Operation(mod))
        finally
            Reactant.MLIR.IR.dispose(mod)
        end
    end
    return operations
end

function _prv_compiled(fx)
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    return Base.invokelatest(_prv_compile_prepared, fx, sampler)
end

function _prv_compile_prepared(fx, sampler)
    ru = Reactant.to_rarray(fx.u)
    kernel, ad = sampler.kernel, sampler.ad
    both(v) = ad_value_and_gradient(ad, v)
    modules = [Reactant.@code_hlo(optimize = false, kernel(ru)),
        Reactant.@code_hlo(kernel(ru)),
        Reactant.@code_hlo(optimize = false, both(ru)),
        Reactant.@code_hlo(both(ru))]
    operations = _prv_operations.(modules)
    if haskey(ENV, "RKPPL_PLATE_RESPONSE_HLO_DIR")
        dir = ENV["RKPPL_PLATE_RESPONSE_HLO_DIR"]
        mkpath(dir)
        for (i, mod) in enumerate(modules)
            write(joinpath(dir, "$(fx.kind)-$(length(fx.data[:y]))-$i.mlir"), String(mod))
        end
    end
    compiled = Reactant.@compile kernel(ru)
    reverse = compile_ad_value_and_gradient(ad, ru)
    for shift in (0.0, 0.03)
        u = fx.u .+ shift
        r = Reactant.to_rarray(u)
        @test Float64(compiled(r)) ≈ fx.oracle(u) rtol = 1e-9
        value, gradient = reverse(r)
        @test Float64(value) ≈ fx.oracle(u) rtol = 1e-9
        @test Array(gradient) ≈ _prv_fd(fx.oracle, u) rtol = 1e-5 atol = 1e-7
    end
    return operations
end

@testset "retained plate response values: default compiled math and structure" begin
    # Warm once at the same dimensions before comparing complete modules.
    Base.invokelatest(_prv_compiled, _prv_fixture(:direct, 3))
    for kind in (:direct, :alias, :alias_literal, :alias_axes, :alias_local,
            :alias_scale, :selected, :bare, :bare_broadcast)
        bare = kind in (:bare, :bare_broadcast)
        indexed = kind !== :bare_broadcast
        small = bare ? _prv_bare(3; indexed) : _prv_fixture(kind, 3)
        # A literal `1:n` states its own extent, so each size lowers anew.
        large = bare ? _prv_bare(9; unbound = small.unbound, indexed) :
            _prv_fixture(kind, 9;
                unbound = kind === :alias_literal ? nothing : small.unbound)
        sop = Base.invokelatest(_prv_compiled, small)
        lop = Base.invokelatest(_prv_compiled, large)
        @test sop[1] == lop[1]
        @test sop[3] == lop[3]
        if bare
            # Released Reactant/XLA expands the tiny guarded Binomial loop
            # at 3 rows, while 9 rows retain control flow. The unchanged
            # broadcast form is the independent control for this backend
            # limitation. Keep both default gates explicit and unresolved.
            @test_broken sop[2] == lop[2]
            @test_broken sop[4] == lop[4]
        elseif kind === :alias_scale
            # A row-varying scale guards each row's log density. Released
            # Reactant/XLA expands that guarded loop at 3 rows and retains
            # one body from 9 rows; the same scale written inline, without
            # reading `mu`, expands identically (backend-only reproducer
            # benchmark/repro_reactant_lazy_batch_growth.jl). Keep the
            # small-size gate explicit and require retention above it.
            @test_broken sop[2] == lop[2]
            @test_broken sop[4] == lop[4]
            xop = Base.invokelatest(_prv_compiled,
                _prv_fixture(kind, 17; unbound = small.unbound))
            @test lop == xop
        else
            @test sop[2] == lop[2]
            @test sop[4] == lop[4]
        end
    end
end
