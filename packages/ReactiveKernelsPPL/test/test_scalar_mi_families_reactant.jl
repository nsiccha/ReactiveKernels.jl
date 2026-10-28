using Reactant

function _scalar_mi_inventory_delta(before,after)
    Dict(op=>get(after,op,0)-get(before,op,0) for op in union(keys(before),keys(after))
        if get(after,op,0)!=get(before,op,0))
end

@testset "scalar Case-A families retain default primal and reverse graphs" begin
    previous=Dict{Any,Any}()
    executable_previous=Dict{Any,Any}()
    # Fourfold packed-row growth (8 to 32). The complete ordinal inventories
    # at 15/31/63 show shape-specific canonicalization at 31, not body growth;
    # both primal and reverse inventories at 63 exactly match those at 15.
    for n in (15,63), (kind,variant) in [[(k,:plain) for k in _SCALAR_MI_KINDS]...,(:stopping,:gathered),(:stopping,:censored)]
        @testset "$kind / $variant / n=$n" begin
            f=_scalar_mi_fixture(kind,n;variant)
            kernel=f.kernel
            ru=Reactant.to_rarray(f.u)
            ad=Base.invokelatest(prepare_ad,kernel,AutoEnzyme(; mode=Enzyme.Reverse),f.u;active=:unconstrained)
            compiled=Reactant.@compile kernel(ru)
            if variant===:censored
                # Generic host-matrix capture repair72ae21f5 lifts primal
                # tracing. Keep real positive sampler and pointwise checks.
                @test Float64(compiled(ru)) ≈ f.oracle(f.u)
                pointwise=prepare_query(f.built,f.bound,:pointwise)
                compiled_pw=Reactant.@compile pointwise(ru)
                @test Array(only(values(compiled_pw(ru)))) ≈ f.pointwise(f.u)
                @test f.data==f.saved
                pair=(_probability_value_inventory(repr(Reactant.@code_hlo kernel(ru)),"censored-$n-sampler"),
                    _probability_value_executable_inventory(compiled,"censored-$n-sampler"),
                    _probability_value_inventory(repr(Reactant.@code_hlo pointwise(ru)),"censored-$n-pointwise"),
                    _probability_value_executable_inventory(compiled_pw,"censored-$n-pointwise"))
                @test all(x->!isempty(x),pair)
                if n==15
                    previous[(kind,variant)]=pair
                else
                    # Data-bound evidence splits into small lazy batches.
                    # Default lowering expands the smaller batch at 15 rows;
                    # at 63 rows both batches retain loops. Numerical parity
                    # does not lift the documented lazy-batch growth limit.
                    mlir_delta=Dict("arith.constant"=>1,"stablehlo.add"=>1,
                        "stablehlo.broadcast_in_dim"=>2,"stablehlo.concatenate"=>-1,
                        "stablehlo.dynamic_slice"=>1,"stablehlo.dynamic_update_slice"=>1,
                        "stablehlo.exponential"=>-2,"stablehlo.if"=>-1,
                        "stablehlo.log_plus_one"=>-2,"stablehlo.negate"=>-2,
                        "stablehlo.reshape"=>-3,"stablehlo.slice"=>-1,
                        "stablehlo.subtract"=>-3,"stablehlo.while"=>1)
                    hlo_delta=Dict("bitcast"=>1,"broadcast"=>3,"call"=>1,
                        "concatenate"=>-1,"constant"=>1,"copy"=>2,
                        "dynamic-slice"=>1,"dynamic-update-slice"=>1,
                        "exponential"=>-2,"fusion"=>6,"get-tuple-element"=>6,
                        "log-plus-one"=>-2,"negate"=>-2,"parameter"=>15,
                        "select"=>-1,"slice"=>-1,"subtract"=>-2,"tuple"=>2,"while"=>1)
                    pointwise_delta=copy(hlo_delta)
                    pointwise_delta["parameter"]=14
                    @test map(_scalar_mi_inventory_delta,previous[(kind,variant)],pair)==
                        (mlir_delta,hlo_delta,mlir_delta,pointwise_delta)
                    @test_broken pair==previous[(kind,variant)]
                end
                # Handler1d8anli independently reproduced DEFAULT reverse
                # SIGABRT134, chained slices8xf64 vs6xf64. Core constraints
                # require skipping an aborting acceptance case by name.
                @test_skip begin
                    reverse=compile_ad_value_and_gradient(ad,ru)
                    value,gradient=reverse(ru)
                    Float64(value) ≈ f.oracle(f.u) &&
                        isapprox(Array(gradient),_distributional_findiff(f.oracle,f.u);rtol=2e-5,atol=2e-7)
                end
                continue
            end
            reverse=compile_ad_value_and_gradient(ad,ru)
            value,gradient=reverse(ru)
            @test Float64(compiled(ru)) ≈ f.oracle(f.u)
            @test Float64(value) ≈ f.oracle(f.u)
            @test Array(gradient) ≈ _distributional_findiff(f.oracle,f.u) rtol=2e-5 atol=2e-7
            @test f.data == f.saved
            both=reverse.f
            pair=(_probability_value_inventory(repr(Reactant.@code_hlo kernel(ru)),"scalar-$kind-$variant-$n-primal"),
                _probability_value_inventory(repr(Reactant.@code_hlo both(ru)),"scalar-$kind-$variant-$n-reverse"))
            @test !isempty(pair[1]) && !isempty(pair[2])
            n==15 ? (previous[(kind,variant)]=pair) : (@test pair==previous[(kind,variant)])
            executable=(_probability_value_executable_inventory(compiled,"scalar-$kind-$variant-$n-primal"),
                _probability_value_executable_inventory(reverse,"scalar-$kind-$variant-$n-reverse"))
            @test !isempty(executable[1]) && !isempty(executable[2])
            if n==15
                executable_previous[(kind,variant)]=executable
            elseif kind===:stopping
                # The complete default executable adds one reduction stage,
                # despite unchanged optimized MLIR and numerical parity.
                # See benchmark/repro_reactant_vector_reduction_growth.jl.
                # Pin both FULL deltas; no operation names are discarded.
                delta=map(_scalar_mi_inventory_delta,
                    executable_previous[(kind,variant)],executable)
                @test delta==(Dict("reduce-window"=>1,"slice"=>1,"constant"=>1,
                    "add"=>1,"fusion"=>4,"bitcast"=>2,"parameter"=>10),
                    Dict("reduce-window"=>1,"add"=>1,"fusion"=>2,"parameter"=>5))
                @test_broken executable==executable_previous[(kind,variant)]
            else
                @test executable==executable_previous[(kind,variant)]
            end
        end
    end
end

@testset "uncensored stopping-ratio pointwise retains default structure" begin
    previous=Dict{Symbol,Any}()
    executable_previous=Dict{Symbol,Any}()
    for n in (15,31), variant in (:plain,:gathered)
        f=_scalar_mi_fixture(:stopping,n;variant)
        kernel=prepare_query(f.built,f.bound,:pointwise)
        ru=Reactant.to_rarray(f.u)
        compiled=Reactant.@compile kernel(ru)
        @test Array(only(values(compiled(ru)))) ≈ f.pointwise(f.u)
        graph=_probability_value_inventory(repr(Reactant.@code_hlo kernel(ru)),"pointwise-$variant-$n-primal")
        @test !isempty(graph)
        n==15 ? (previous[variant]=graph) : (@test graph==previous[variant])
        executable=_probability_value_executable_inventory(compiled,"pointwise-$variant-$n-primal")
        @test !isempty(executable)
        if n==15
            executable_previous[variant]=executable
        else
            # Cumulative pointwise reduction adds a default XLA stage too.
            # Preserve the complete delta and the unmet inventory assertion.
            @test _scalar_mi_inventory_delta(executable_previous[variant],executable)==
                Dict("add"=>2,"bitcast"=>3,"broadcast"=>1,"fusion"=>2,
                    "pad"=>1,"parameter"=>6,"reduce-window"=>1,"slice"=>3)
            @test_broken executable==executable_previous[variant]
        end
    end
end
