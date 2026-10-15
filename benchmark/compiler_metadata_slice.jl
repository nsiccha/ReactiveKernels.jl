# Run unchanged in separate baseline/candidate environments and fresh Julia
# processes. Rows distinguish first use in the process from later warm cases.
# No benchmark package or private model data is needed.
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Distributions: Normal, logpdf
using Test, TOML

metadata_ast_nodes(x) = x isa Expr ? 1 + sum(metadata_ast_nodes, x.args; init=0) : 1
metadata_timing(t) = (; seconds=t.time, bytes=t.bytes, compile_seconds=t.compile_time)

function metadata_slice_case(n, width)
    predictor = :(alpha .+ beta .* x .+ b[group])
    for _ in 2:width
        predictor = :($predictor .+ beta .* x)
    end
    source = quote
        alpha ~ Normal(0, 2)
        beta ~ Normal(0, 2)
        b[1:3] .~ Normal.(0, 1)
        mu = $predictor
        y .~ Normal.(mu, 1)
    end
    x = [sin(i) for i in 1:n]
    group = [mod1(i, 3) for i in 1:n]
    y = [cos(i) for i in 1:n]
    data = (; x, group, y)
    saved = deepcopy(data)
    construction = @timed bind_data(lower_rkppl(source, data;
        conditioned=(:y,), mod=@__MODULE__), data)
    plan = construction.value
    build = @timed build_kernel(plan)
    built = build.value
    alpha, beta, b = 0.2, -0.3, [-0.2, 0.1, 0.3]
    u = collect(unconstrain(built.layout, (; alpha, beta, b)))
    preparation = @timed prepare_query(built, plan, :sampler)
    q = preparation.value
    primal = @timed q(u)
    ad_preparation = @timed prepare_sampler(built, plan, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    sampler = ad_preparation.value
    gradient = similar(u)
    derivative = @timed sampler_value_and_gradient!(sampler, gradient, u)
    mu = alpha .+ width .* beta .* x .+ b[group]
    expected = logpdf(Normal(0, 2), alpha) + logpdf(Normal(0, 2), beta) +
        sum(logpdf.(Normal(), b)) + sum(logpdf.(Normal(), y .- mu))
    @test primal.value ≈ expected rtol=1e-12
    @test first(derivative.value) ≈ expected rtol=1e-12
    for i in eachindex(u)
        up, down = copy(u), copy(u)
        up[i] += 1e-6
        down[i] -= 1e-6
        @test gradient[i] ≈ (q(up)-q(down))/2e-6 rtol=1e-6 atol=1e-6
    end
    @test data == saved
    emitted = kernel_expr(plan, built.layout)
    row = (; n, width, construction=metadata_timing(construction),
        build=metadata_timing(build), preparation=metadata_timing(preparation),
        first_primal=metadata_timing(primal),
        ad_preparation=metadata_timing(ad_preparation),
        first_derivative=metadata_timing(derivative),
        emitted_nodes=metadata_ast_nodes(emitted),
        emitted_bytes=sizeof(sprint(show, Base.remove_linenums!(deepcopy(emitted)))))
    println(row)
    flush(stdout)
    return row
end

rows = Any[]
@testset "compiler metadata migration: values and ordinary Reverse" begin
    for width in (1, 4), n in (8, 256)
        push!(rows, metadata_slice_case(n, width))
    end
end
if !isempty(ARGS)
    open(ARGS[1], "w") do io
        TOML.print(io, Dict("julia"=>string(VERSION),
            "rows"=>[Dict(string(k)=>v isa NamedTuple ? Dict(string(a)=>b for (a,b) in pairs(v)) : v
                for (k,v) in pairs(row)) for row in rows]))
    end
end
