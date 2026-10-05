using Reactant

function _missing_save_ir(name, kind, text)
    haskey(ENV, "RK_PPL_MISSING_IR_DIR") || return
    dir = ENV["RK_PPL_MISSING_IR_DIR"]
    mkpath(dir)
    write(joinpath(dir, "$name-$kind"), text)
end

function _missing_compiled_check(fx, name; structure=false)
    println("MISSING_COMPILED ", name)
    saved = deepcopy(fx.data)
    kernel = fx.sampler.kernel
    ru = Reactant.to_rarray(fx.u)
    primal = Reactant.@compile kernel(ru)
    reverse = compile_ad_value_and_gradient(fx.sampler.ad, ru)
    for u in (fx.u, fx.u .+ 0.15)
        input = Reactant.to_rarray(u)
        value, grad = sampler_value_and_gradient!(fx.sampler, similar(u), u)
        cv, cg = reverse(input)
        @test Float64(primal(input)) ≈ fx.oracle(u) rtol=1e-9
        @test Float64(cv) ≈ value rtol=1e-9
        @test Array(cg) ≈ grad rtol=1e-8 atol=1e-9
        @test Array(input) == u
    end
    query = prepare_query(fx.built, fx.bound, :pointwise)
    compiled_query = Reactant.@compile query(ru)
    native = Base.invokelatest(query, fx.u)
    actual = compiled_query(ru)
    @test Array(actual.y) ≈ native.y
    @test size(Array(actual.y)) == size(fx.data.y)
    @test isequal(fx.data, saved)
    if structure
        texts = ("primal.mlir" => repr(Reactant.@code_hlo kernel(ru)),
            "reverse.mlir" => repr(Reactant.@code_hlo reverse.f(ru)),
            "primal.hlo" => repr(only(Reactant.XLA.get_hlo_modules(primal.exec))),
            "reverse.hlo" => repr(only(Reactant.XLA.get_hlo_modules(reverse.exec))))
        for (kind, text) in texts
            _missing_save_ir(name, kind, text)
            ops = Dict{String,Int}()
            regex = endswith(kind, "mlir") ? r"stablehlo\.[a-z_]+" :
                r"(?m)^\s*(?:ROOT )?%[\w.\-]+ = .*? ([a-z][a-z0-9-]*)\("
            for m in eachmatch(regex, text)
                op = endswith(kind, "mlir") ? m.match : m.captures[1]
                ops[op] = get(ops, op, 0) + 1
            end
            _missing_save_ir(name, "$kind.inventory", repr(sort!(collect(ops))))
            println("MISSING_STRUCTURE ", name, " ", kind, " ", sort!(collect(ops)))
        end
    end
end

@testset "Reactant: automatic missing observations, ordinary AD and retained batching" begin
    for kind in (:vector, :computed_plate, :matrix), n in (5, 9)
        y = Union{Missing,Float64}[isodd(i) ? 0.03*i : missing for i in 1:n]
        kind === :matrix && (y = hcat(y, reverse(y)))
        fx = _missing_fixture(y; computed=kind === :computed_plate)
        _missing_compiled_check(fx, "$kind-$n"; structure=true)
    end
    for y in (Union{Missing,Float64}[], fill(missing, 5))
        _missing_compiled_check(_missing_fixture(y), "empty-or-all-$(length(y))")
    end
    for kind in (:binomial, :bernoulli, :beta, :stopping, :invalid_missing_scale)
        _missing_compiled_check(_missing_family_fixture(kind), "family-$kind")
    end
end
