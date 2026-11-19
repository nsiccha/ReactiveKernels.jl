using DifferentiationInterface, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

module LoweringCacheRows
const calls = Ref(0)
function rows(groups, cells, mode)
    calls[] += 1
    mode < 0 && error("invalid partition mode")
    mode == 0 && return collect(eachindex(cells))
    return [findall(==(c), groups) for c in cells]
end
end

@testset "value-aware lowering rechecks computed indices on hits and misses" begin
    ast = quote
        a ~ Normal(0, 1)
        theta = a .+ shift
        indices = rows(g, cells, mode)
        @plate for i in eachindex(cells)
            loc[i] = x[indices[i]] * theta[i]
        end
        @plate for i in eachindex(y)
            y[i] .~ Normal.(loc[i], 1)
        end
    end
    d = (; g=["a","b","b","a","a"], cells=["b","empty","a"],
        x=[0.2,0.5,0.8,1.1,1.4], shift=[-0.2,0.05,0.3],
        y=[[0.1,0.4], Float64[], [-0.2,0.33,0.8]], mode=0)
    cache = LoweringCache()
    lower(data, cache) = lower_rkppl(ast, data; mod=LoweringCacheRows, conditioned=(:y,), cache)
    scalar = lower(d, cache)
    @test length(cache.entries) == 1
    arrays = merge(d, (; mode=1))
    @test typeof(d) === typeof(arrays)
    more = merge(arrays, (; g=[d.g; "c"; "a"], cells=[d.cells; "c"],
        x=[d.x; -0.6; 0.4], shift=[d.shift; -0.1],
        y=[[0.1,0.4], Float64[], [-0.2,0.33,0.8,0.9], [0.4]]))
    programs = Expr[]
    for current in (arrays, more)
        before = LoweringCacheRows.calls[]
        plan = lower(current, cache)
        @test LoweringCacheRows.calls[] == before + 1
        @test length(cache.entries) == 2
        bound = bind_data(plan, current)
        fresh = bind_data(lower(current, nothing), current)
        built = build_kernel(bound)
        @test built.program == build_kernel(fresh).program
        push!(programs, built.program)
        u = [0.1]
        sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        expected = -0.5 * (log(2pi) + only(u)^2)
        derivative = -only(u)
        for (i, cell) in pairs(current.cells)
            x = current.x[findall(==(cell), current.g)]
            residual = current.y[i] .- x .* (only(u) + current.shift[i])
            expected -= (length(x)*log(2pi) + sum(abs2, residual))/2
            derivative += sum(residual .* x)
        end
        @test value ≈ expected
        @test only(grad) ≈ derivative
    end
    @test allequal(programs)
    @test_throws ContractValidationError bind_data(scalar, arrays)
    @test_throws "invalid partition mode" lower(merge(d, (; mode=-1)), cache)
    @test length(cache.entries) == 2
end

@testset "current source and visible callables choose a lowering" begin
    cache = LoweringCache()
    ast = quote
        center = summarize(x)
        a ~ Normal(0, 1)
        mu = a .+ (x .- center)
        y .~ Normal.(mu, 1)
    end
    data = (; x=[2., 3., 8.], y=zeros(3))
    mod = Module(gensym(:Fit))
    lower(caller=mod) = lower_rkppl(ast, data; mod=caller, conditioned=(:y,), cache)
    center(plan) = frozen_values(bind_data(plan, data), :center).values.center
    Core.eval(mod, :(summarize = minimum))
    first_plan = lower()
    @test center(first_plan) == 2
    empty!(first_plan.responses)
    Core.eval(mod, :(summarize = maximum))
    other = Module(gensym(:Fit))
    Core.eval(other, :(summarize = minimum))
    # The previous module now computes 8. A hit must call the current module.
    replay = lower(other)
    @test length(cache.entries) == 1
    @test !isempty(replay.responses) && center(replay) == 2
    @test center(lower()) == 8
    @test length(cache.entries) == 2
    # A closure can refer to an entire fit. It must not enter the cache key.
    Core.eval(mod, :(summarize = let offset = [10.]
        x -> minimum(x) + only(offset)
    end))
    @test center(lower()) == 12
    @test length(cache.entries) == 2
    Core.eval(mod, :(summarize = minimum))
    index = findfirst(st -> Meta.isexpr(st, :(=)) && st.args[1] === :center, ast.args)
    ast.args[index] = :(center = summarize(x) + 1)
    @test center(lower()) == 3
    @test length(cache.entries) == 3
end

@testset "a downstream image replays a fitted lowering in a new Julia process" begin
    # A package directory lets the child use the active test environment and
    # its exact dependency versions, without a resolver or network access.
    fixture = joinpath(@__DIR__, "fixtures")
    load_path = join([fixture; Base.load_path()], Sys.iswindows() ? ';' : ':')
    probe = joinpath(fixture, "replay_lowering.jl")
    child = addenv(`$(Base.julia_cmd()) --startup-file=no --threads=2 $probe`, "JULIA_LOAD_PATH" => load_path)
    @test success(pipeline(child; stdout, stderr))
end
