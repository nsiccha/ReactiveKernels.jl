using Reactant

function _probability_value_inventory(hlo,label=nothing)
    if label !== nothing && haskey(ENV,"RK_PPL_ACCEPTANCE_HLO_DIR")
        write(joinpath(ENV["RK_PPL_ACCEPTANCE_HLO_DIR"],"$label.mlir"),hlo)
    end
    counts=Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.\w+",hlo)
        counts[m.match]=get(counts,m.match,0)+1
    end
    counts
end

function _probability_value_executable_inventory(compiled,label)
    hlo=repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
    counts=Dict{String,Int}()
    for m in eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(",hlo)
        name=m.captures[1]
        counts[name]=get(counts,name,0)+1
    end
    if haskey(ENV,"RK_PPL_ACCEPTANCE_HLO_DIR")
        write(joinpath(ENV["RK_PPL_ACCEPTANCE_HLO_DIR"],"$label.hlo"),hlo)
    end
    println("EXECUTABLE_INVENTORY ",label," ",sort!(collect(counts);by=first))
    return counts
end

@testset "probability arithmetic retains default primal and reverse graphs" begin
    previous=Dict{Any,Any}()
    executable_previous=Dict{Any,Any}()
    for n in (15,31), family in (:Bernoulli,:Binomial), named in (false,true), scalar in (false,true)
        f=_probability_value_fixture(family,n; named,scalar)
        kernel=f.kernel
        ru=Reactant.to_rarray(f.u)
        ad=Base.invokelatest(prepare_ad,kernel,AutoEnzyme(; mode=Enzyme.Reverse),f.u;active=:unconstrained)
        compiled=Reactant.@compile kernel(ru)
        reverse=compile_ad_value_and_gradient(ad,ru)
        value,gradient=reverse(ru)
        @test Float64(compiled(ru)) ≈ f.oracle(f.u)
        @test Float64(value) ≈ f.oracle(f.u)
        @test Array(gradient) ≈ _distributional_findiff(f.oracle,f.u) rtol=2e-5 atol=2e-7
        both=reverse.f
        pair=(_probability_value_inventory(repr(Reactant.@code_hlo kernel(ru)),"probability-$family-$named-$scalar-$n-primal"),
            _probability_value_inventory(repr(Reactant.@code_hlo both(ru)),"probability-$family-$named-$scalar-$n-reverse"))
        @test !isempty(pair[1]) && !isempty(pair[2])
        key=(family,named,scalar)
        n==15 ? (previous[key]=pair) : (@test pair==previous[key])
        executable=(_probability_value_executable_inventory(compiled,"probability-$family-$named-$scalar-$n-primal"),
            _probability_value_executable_inventory(reverse,"probability-$family-$named-$scalar-$n-reverse"))
        @test !isempty(executable[1]) && !isempty(executable[2])
        n==15 ? (executable_previous[key]=executable) : (@test executable==executable_previous[key])
    end
end

@testset "invalid probability values and gradients with executable diagnostics" begin
    data=(; x=[-0.4,0.1,0.6],y=[false,true,true])
    f=_distributional_model(quote
        a ~ Normal(0,1)
        b ~ Normal(0,1)
        eta = a .+ b .* x
        y .~ Bernoulli.(eta .+ 0.1)
    end,data)
    kernel=f.kernel
    u=[-0.5,0.0]
    ru=Reactant.to_rarray(u)
    ad=Base.invokelatest(prepare_ad,kernel,AutoEnzyme(;mode=Enzyme.Reverse),u;active=:unconstrained)
    compiled=Reactant.@compile kernel(ru)
    reverse=compile_ad_value_and_gradient(ad,ru)
    value,gradient=reverse(ru)
    @test Float64(compiled(ru)) == -Inf
    @test Float64(value) == -Inf
    @test Array(gradient) ≈ -u
    @test data == (; x=[-0.4,0.1,0.6],y=[false,true,true])
    primal_ops=_probability_value_executable_inventory(compiled,"probability-invalid-primal")
    reverse_ops=_probability_value_executable_inventory(reverse,"probability-invalid-reverse")
    # Stock XLA may speculate pure floating-point log/divide before a select.
    # The required result is -Inf with the finite prior gradient above, not
    # a prescribed conditional/log/select count (docs/src/constraints.md).
    @test !isempty(primal_ops)
    @test !isempty(reverse_ops)
end
