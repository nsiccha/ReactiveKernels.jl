using Reactant

function _cgi_operations(module_text)
    names = String[]
    function walk(op)
        push!(names, Reactant.MLIR.IR.name(op))
        for region in op, block in region, child in block
            walk(child)
        end
    end
    Reactant.MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        mod = parse(Reactant.MLIR.IR.Module, String(module_text); context=ctx)
        try
            walk(Reactant.MLIR.IR.Operation(mod))
        finally
            Reactant.MLIR.IR.dispose(mod)
        end
    end
    return names
end

function _cgi_compiled(fx)
    q = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    # Preparation can eval new query/branch methods. Enter their world before
    # tracing, keeping the barrier outside both compiled callables.
    return Base.invokelatest(_cgi_compile_prepared, fx, q)
end

function _cgi_compile_prepared(fx, q)
    ru = Reactant.to_rarray(fx.u)
    primal = q.kernel
    ad = q.ad
    both(v) = ad_value_and_gradient(ad, v)
    modules = [Reactant.@code_hlo(optimize=false, primal(ru)),
        Reactant.@code_hlo(primal(ru)),
        Reactant.@code_hlo(optimize=false, both(ru)),
        Reactant.@code_hlo(both(ru))]
    # Compare every dialect's ordered operations, including packing and
    # control-flow operations; keep the full modules as optional receipts.
    operations = _cgi_operations.(modules)
    if haskey(ENV, "RKPPL_COMPUTED_GATHER_HLO_DIR")
        dir = ENV["RKPPL_COMPUTED_GATHER_HLO_DIR"]
        mkpath(dir)
        for (i, hlo) in enumerate(modules)
            write(joinpath(dir, "$(fx.kind)-$(length(fx.data[:g]))-$i.mlir"), String(hlo))
        end
    end
    compiled = Reactant.@compile primal(ru)
    reverse = compile_ad_value_and_gradient(q.ad, ru)
    calls = ComputedGatherModels.calls[]
    for shift in (0.0, 0.03)
        u = fx.u .+ shift
        r = Reactant.to_rarray(u)
        oracle(w) = _cgi_oracle(fx, w).value
        @test Float64(compiled(r)) ≈ oracle(u) rtol=1e-9
        value, gradient = reverse(r)
        @test Float64(value) ≈ oracle(u) rtol=1e-9
        @test Array(gradient) ≈ _cgi_findiff(oracle, u) rtol=1e-5 atol=1e-7
    end
    @test ComputedGatherModels.calls[] == calls
    return operations
end

@testset "computed gather indices: default compiled reverse and fixed structure" begin
    # First preparation can retain a different fixed raw batch wrapper. A
    # second model at the SAME dimensions chooses the subsequent template:
    # this is initialization state, not data-size expansion. Check that cold
    # model numerically and retain all four modules, then compare full raw
    # and optimized inventories after initialization. Its optimized primal
    # and reverse inventories must also match the subsequent model's.
    cold = Base.invokelatest(_cgi_compiled, _cgi_build(:bootstrap, 2, 6))
    for kind in (:named, :second, :inline, :opaque, :levels)
        fx = _cgi_build(kind, 2, 6)
        small = Base.invokelatest(_cgi_compiled, fx)
        large = Base.invokelatest(_cgi_compiled,
            _cgi_build(kind, 5, 18; unbound=fx.unbound))
        @test small == large
        if kind === :named
            @test cold[2] == small[2]
            @test cold[4] == small[4]
        end
    end
    Base.invokelatest(_cgi_compiled, _cgi_build(:named, 0, 0))
end
