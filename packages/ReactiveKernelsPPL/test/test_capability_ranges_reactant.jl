using Reactant

function _cap_range_operations(hlo)
    out = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        out[m.match] = get(out, m.match, 0) + 1
    end
    out
end

function _cap_range_xla_operations(hlo)
    out = Dict{String,Int}()
    for m in eachmatch(r"(?m)^\s*(?:ROOT )?%[\w.\-]+ = .*? ([a-z][a-z0-9-]*)\(", hlo)
        op = m.captures[1]
        out[op] = get(out, op, 0) + 1
    end
    out
end

function _cap_range_save_ir(kind, n, name, text)
    haskey(ENV, "RK_PPL_RANGE_IR_DIR") || return nothing
    dir = ENV["RK_PPL_RANGE_IR_DIR"]
    mkpath(dir)
    write(joinpath(dir, "$kind-$n-$name"), text)
    return nothing
end

@testset "Reactant: response slices and retained selected cells" begin
    for kind in (:cross_index, :cross_axis, :colon, :cross_cell, :free_matrix, :longer_inactive)
        primal, reverse = Dict{String,Int}[], Dict{String,Int}[]
        primal_xla, reverse_xla = Dict{String,Int}[], Dict{String,Int}[]
        # Compare the optimized programs and actual executables. Raw tracing
        # can share shape-dependent constants and scalar broadcasts; exact
        # raw inventory equality is not the retained-batching requirement.
        for n in (5, 9)
            println("RANGE_COMPILED kind=", kind, " n=", n)
            fx = _cap_range_fixture(kind, n)
            saved = deepcopy(fx.data)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.sampler.kernel
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ fx.oracle(fx.u)
            value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
            cad = compile_ad_value_and_gradient(fx.sampler.ad, ru)
            cv, cg = cad(ru)
            @test Float64(cv) ≈ value
            @test Array(cg) ≈ gradient rtol=1e-8
            # Preserve the actual default executables alongside optimized
            # MLIR; raw inventory equality alone does not inspect XLA work.
            both = cad.f
            pm = repr(Reactant.@code_hlo kernel(ru))
            rm = repr(Reactant.@code_hlo both(ru))
            ph = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
            rh = repr(only(Reactant.XLA.get_hlo_modules(cad.exec)))
            push!(primal, _cap_range_operations(pm))
            push!(reverse, _cap_range_operations(rm))
            push!(primal_xla, _cap_range_xla_operations(ph))
            push!(reverse_xla, _cap_range_xla_operations(rh))
            for (name, text) in (
                ("primal.mlir", pm), ("reverse.mlir", rm),
                ("primal.hlo", ph), ("reverse.hlo", rh))
                _cap_range_save_ir(kind, n, name, text)
            end
            @test Array(ru) == fx.u
            @test isequal(fx.data, saved)
            pointwise = prepare_query(fx.built, fx.bound, :pointwise)
            cpw = Reactant.@compile pointwise(ru)
            output = cpw(ru)
            if kind === :free_matrix
                @test output == (;)
            else
                @test Array(output.y) ≈ fx.pointwise(fx.u)
            end
        end
        @test primal[1] == primal[2]
        @test reverse[1] == reverse[2]
        @test primal_xla[1] == primal_xla[2]
        @test reverse_xla[1] == reverse_xla[2]
    end
end

@testset "Reactant: empty indexed observations retain prior gradients" begin
    for kind in (:cross_index, :cross_axis, :colon, :cross_cell, :matrix, :free_matrix,
            :literal_tail, :literal_whole, :literal_matrix)
        fx = _cap_range_fixture(kind, 0)
        ru = Reactant.to_rarray(fx.u)
        cad = compile_ad_value_and_gradient(fx.sampler.ad, ru)
        value, gradient = cad(ru)
        @test Float64(value) ≈ fx.oracle(fx.u)
        @test Array(gradient) ≈ _cap_range_fd(fx.oracle, fx.u) rtol=5e-6
    end
end

@testset "empty authored subset cannot skip nonempty missing data" begin
    data = (; y = fill!(Vector{Union{Missing,Float64}}(undef, 6), missing),
        x = fill(-1.0, 6))
    saved = deepcopy(data)
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in 1:0
            y[i] ~ Normal(a + b * sqrt(x[i]), 0.7)
        end
    end
    @test_throws ContractValidationError bind_data(lower_rkppl(ast, data; conditioned = keys(data)), data)
    @test isequal(data, saved)
end
