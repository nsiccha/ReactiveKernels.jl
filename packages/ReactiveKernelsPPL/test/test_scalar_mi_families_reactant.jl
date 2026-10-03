using Reactant

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
            compiled=try
                Reactant.@compile kernel(ru)
            catch err
                # reactant-host-ma-09a51af4 also reproduces on published
                # full-row/no-MI source. Pin only this host-matrix branch ABI
                # failure; other exceptions remain errors. Native evidence
                # and the uncensored matrix variant have positive coverage.
                frames=stacktrace(catch_backtrace())
                known=kind===:stopping && variant===:censored &&
                    err isa BoundsError && err.a isa Matrix{Float64} &&
                    size(err.a)==(length(f.data[:Jobs]),2) &&
                    endswith(sprint(showerror,err),"at index [1]") &&
                    any(frame->frame.func===:traced_getfield &&
                        endswith(string(frame.file),"Codegen.jl"),frames)
                known || rethrow()
                @test known
                @test_broken false
                continue
            end
            reverse=compile_ad_value_and_gradient(ad,ru)
            value,gradient=reverse(ru)
            @test Float64(compiled(ru)) ≈ f.oracle(f.u)
            @test Float64(value) ≈ f.oracle(f.u)
            @test Array(gradient) ≈ _distributional_findiff(f.oracle,f.u) rtol=2e-5 atol=2e-7
            @test f.data == f.saved
            both=reverse.f
            pair=(_probability_value_inventory(repr(Reactant.@code_hlo kernel(ru))),
                _probability_value_inventory(repr(Reactant.@code_hlo both(ru))))
            @test !isempty(pair[1]) && !isempty(pair[2])
            n==15 ? (previous[(kind,variant)]=pair) : (@test pair==previous[(kind,variant)])
            executable=(_probability_value_executable_inventory(compiled,"scalar-$kind-$variant-$n-primal"),
                _probability_value_executable_inventory(reverse,"scalar-$kind-$variant-$n-reverse"))
            @test !isempty(executable[1]) && !isempty(executable[2])
            n==15 ? (executable_previous[(kind,variant)]=executable) :
                (@test executable==executable_previous[(kind,variant)])
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
        graph=_probability_value_inventory(repr(Reactant.@code_hlo kernel(ru)))
        @test !isempty(graph)
        n==15 ? (previous[variant]=graph) : (@test graph==previous[variant])
        executable=_probability_value_executable_inventory(compiled,"pointwise-$variant-$n-primal")
        @test !isempty(executable)
        n==15 ? (executable_previous[variant]=executable) : (@test executable==executable_previous[variant])
    end
end
