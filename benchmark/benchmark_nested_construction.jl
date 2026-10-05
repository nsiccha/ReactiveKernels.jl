# Public, independently synthetic construction example. No private model/data.
# This tests construction machinery and toy algebra, not PK-model acceptance.
# Run in a project providing ReactiveKernels and ReactiveKernelsPPL:
# julia --startup-file=no --project=PROJECT public_construction.jl NEW_OUTPUT
using TOML, Dates, SHA
length(ARGS) == 1 || error("Usage: public_construction.jl NEW_OUTPUT")
output = abspath(only(ARGS))
ispath(output) && error("Refusing to overwrite an earlier run: $output")
mkpath(output)
result = Dict{String,Any}(
    "scope" => "Public synthetic nested plate/scan construction; not private-model acceptance",
    "started_utc" => string(now(UTC)),
    "julia_version" => string(VERSION),
    "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
    "phases" => Any[],
)
save() = open(io -> TOML.print(io, result), joinpath(output, "result.toml"), "w")
function measure(f, name)
    println("PUBLIC_START ", name); flush(stdout)
    row = Dict{String,Any}("name" => name, "status" => "running")
    push!(result["phases"], row); save()
    try
        measurement = @timed f()
        row["seconds"] = measurement.time
        row["allocated_bytes"] = measurement.bytes
        row["gc_seconds"] = measurement.gctime
        row["status"] = "passed"
        println("PUBLIC_FINISHED ", name, " seconds=", measurement.time,
                " allocated_bytes=", measurement.bytes); flush(stdout)
        return measurement.value
    catch ex
        row["status"] = "failed"
        row["error"] = sprint(showerror, ex)
        open(joinpath(output, name * "-error.txt"), "w") do io
            showerror(io, ex, catch_backtrace()); println(io)
        end
        rethrow()
    finally
        save()
    end
end
measure("load_public_packages") do
    Core.eval(Main, :(using ReactiveKernels, ReactiveKernelsPPL))
end
result["rk_commit"] = readchomp(`git -C $(pkgdir(ReactiveKernels)) rev-parse HEAD`)
result["rkppl_commit"] = readchomp(`git -C $(pkgdir(ReactiveKernelsPPL)) rev-parse HEAD`)
measure("author_public_kernels") do
    Core.eval(Main, quote
        @kernel public_recurrence(xs, gain) = begin
            states = scan(xs, Ref(gain); init=0.0) do previous, x, g
                next = previous + x * g
                (next, next)
            end
            return states
        end
        @kernel public_adapter(xs, gain) = begin
            states = public_recurrence(xs, gain)
            return states
        end
        @kernel public_population(subjects, gain) = begin
            rows = plate(subjects, Ref(gain)) do xs, g
                states = public_adapter(xs, g)
                states
            end
            locations = reduce(vcat, rows)
            return locations
        end
    end)
end
const program = quote
    gain ~ Normal(0.0, 1.0)
    locations = public_population(subjects, gain)
    y .~ Normal.(locations, 1.0)
end
const data = (; subjects=[[0.2, 0.4], [0.1, -0.3, 0.5]], y=zeros(5))
const original_data = deepcopy(data)
write(joinpath(output, "program.jl"), sprint(Base.show_unquoted, program))
open(io -> TOML.print(io, Dict("subjects" => data.subjects, "y" => data.y)),
     joinpath(output, "synthetic-data.toml"), "w")
rkppl_plan = measure("first_lower_rkppl") do
    Base.invokelatest(lower_rkppl, program, data; conditioned=(:y,), mod=Main)
end
bound = measure("first_bind_data") do
    Base.invokelatest(bind_data, rkppl_plan, data)
end
built = measure("first_build_kernel") do
    Base.invokelatest(build_kernel, bound)
end
function inventory(graph, depth=0)
    rows = Any[]
    for recipe in graph.recipes
        for (kind, accessor) in (("plate", plate_body), ("scan", scan_body))
            body = try
                accessor(recipe)
            catch ex
                ex isa ArgumentError || rethrow()
                nothing
            end
            body === nothing && continue
            push!(rows, Dict("kind" => kind, "depth" => depth))
            append!(rows, inventory(body, depth+1))
            break
        end
    end
    rows
end
measure("inspect_built_model") do
    view = Base.invokelatest(model_view, built)
    write(joinpath(output, "built-source.jl"), sprint(show, MIME"text/plain"(), view.code))
    write(joinpath(output, "built-graph.txt"), sprint(show, MIME"text/plain"(), view.graph))
    result["coordinates"] = built.layout.total
    result["inventory"] = inventory(kernel_graph(built.spec))
    @assert any(row["kind"] == "plate" for row in result["inventory"])
    @assert any(row["kind"] == "scan" && row["depth"] >= 1 for row in result["inventory"])
end
query = measure("prepare_toy_sampler") do
    Base.invokelatest(prepare_query, built, bound, :sampler)
end
measure("toy_density_and_caller_data_check") do
    gain = 0.8
    value = Base.invokelatest(query, [gain])
    locations = reduce(vcat, [gain .* cumsum(xs) for xs in data.subjects])
    expected = -0.5 * log(2pi) - 0.5 * gain^2 +
               sum(-0.5 * log(2pi) .- 0.5 .* (data.y .- locations).^2)
    result["toy_density"] = value
    result["independent_toy_density"] = expected
    @assert isapprox(value, expected; rtol=1e-12, atol=1e-12)
    @assert isequal(data, original_data)
    result["caller_data_unchanged"] = true
end
warm_plan = measure("repeat_lower_rkppl") do
    Base.invokelatest(lower_rkppl, program, data; conditioned=(:y,), mod=Main)
end
warm_bound = measure("repeat_bind_data") do
    Base.invokelatest(bind_data, warm_plan, data)
end
warm_built = measure("repeat_build_kernel") do
    Base.invokelatest(build_kernel, warm_bound)
end
@assert warm_built.layout.total == built.layout.total
result["ended_utc"] = string(now(UTC))
result["status"] = "passed"
save()
cp(@__FILE__, joinpath(output, "public_construction.jl"))
println("PUBLIC_COMPLETE ", result["rk_commit"], " coordinates=", built.layout.total)
