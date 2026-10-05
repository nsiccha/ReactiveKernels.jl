using Reactant

function _check_prior_selection_compiled(fx)
    prior, iterator, n = fx.prior, fx.iterator, fx.n
    nobs = length(fx.inputs.y)
    saved = deepcopy(fx.inputs)
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    kernel = sampler.kernel
    reverse(v) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
    ru = Reactant.to_rarray(fx.u)
    for (direction, fn) in ((:primal, kernel), (:reverse, reverse))
        emitted = repr(Reactant.@code_hlo optimize=false fn(ru))
        mlir = repr(Reactant.@code_hlo optimize=:all fn(ru))
        compiled = Reactant.@compile optimize=:all fn(ru)
        hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
        for shift in (0.0, 0.03)
            u = fx.u .+ shift
            input = Reactant.to_rarray(u)
            if direction === :primal
                @test Float64(compiled(input)) ≈ fx.oracle(u)
            else
                _, expected = sampler_value_and_gradient!(sampler, similar(u), u)
                @test Array(compiled(input)) ≈ expected rtol=1e-8 atol=1e-9
            end
            @test Array(input) == u
        end
        println("PRIOR_SELECTION_IR prior=", prior, " iterator=", iterator,
            " n=", n, " nobs=", nobs, " direction=", direction,
            " emitted_while=", count("stablehlo.while", emitted),
            " emitted_bytes=", ncodeunits(emitted))
        if haskey(ENV, "RKPPL_PRIOR_SELECTION_IR_DIR")
            dir = ENV["RKPPL_PRIOR_SELECTION_IR_DIR"]
            mkpath(dir)
            write(joinpath(dir, "$prior-$iterator-$n-$nobs-$direction.emitted.mlir"), emitted)
            write(joinpath(dir, "$prior-$iterator-$n-$nobs-$direction.mlir"), mlir)
            write(joinpath(dir, "$prior-$iterator-$n-$nobs-$direction.hlo"), hlo)
        end
    end
    @test Array(ru) == fx.u
    @test fx.inputs == saved
    return nothing
end

@testset "Reactant: selected prior values and ordinary reverse" begin
    for prior in (:data, :active, :mapped), (n, nobs) in ((0, 4), (1, 4), (6, 4), (18, 11)),
            iterator in (:eachindex, :value)
        _check_prior_selection_compiled(_prior_selection_fixture(n, nobs; prior, iterator))
    end
end

@testset "Reactant: bound parameter and computed iterator extents" begin
    for iterator in (:parameter, :parameter_axis),
            (n, nobs) in ((0, 4), (1, 4), (6, 4), (18, 11))
        _check_prior_selection_compiled(_prior_selection_fixture(n, nobs;
            prior=:active, iterator))
    end
    for iterator in (:scan, :scan_alias, :plate, :opaque_data),
            (n, nobs) in ((1, 4), (6, 4), (18, 11))
        _check_prior_selection_compiled(_iterator_extent_fixture(n, nobs; iterator))
    end
    for iterator in (:plate, :opaque_data)
        _check_prior_selection_compiled(_iterator_extent_fixture(0, 4; iterator))
    end
end
