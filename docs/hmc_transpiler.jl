module HMCTranspilerDocs
using Markdown, SHA, Statistics, TOML, Printf

const DIR = normpath(joinpath(@__DIR__, "..", "benchmark", "sampler_transpiler"))

render_source(filename="position_multinomial_hmc_kernel.jl") =
    Markdown.MD(Markdown.Code("julia", read(joinpath(DIR, filename), String)))

function render_results()
    receipt = TOML.parsefile(joinpath(DIR, "position-multinomial-scaling-v2.toml"))
    data = read(joinpath(DIR, "position-multinomial-scaling-v2.csv"), String)
    bytes2hex(sha256(data)) == receipt["data_sha256"] || error("HMC scaling data changed")
    lines = split(chomp(data), '\n')
    headers = split(first(lines), ',')
    rows = [Dict(zip(headers, split(line, ','))) for line in lines[2:end]]
    execute = filter(row -> row["phase"] == "execute", rows)
    length(execute) == receipt["execute_rows"] || error("HMC execution rows missing")
    for row in execute
        parse(Int, row["gradient_evaluations"]) ==
            parse(Int, row["steps_per_transition"]) * parse(Int, row["transitions"]) ||
            error("HMC integration-work axis mismatch")
    end
    io = IOBuffer()
    println(io, "| Steps | Native median | Reactant median | AdvancedHMC fastest sample |")
    println(io, "| --- | ---: | ---: | ---: |")
    for steps in (4, 16)
        selected = filter(row -> row["steps_per_transition"] == string(steps) &&
                                  row["transitions"] == "10000", execute)
        times(backend) = [parse(Float64, row["seconds"]) * 1e6 / 10000
                          for row in selected if row["backend"] == backend]
        native, reactant, ahmc = times("native"), times("Reactant"), times("AdvancedHMC")
        length(native) == length(reactant) == receipt["replicates"] ||
            error("HMC replicates missing")
        length(ahmc) == 2 * receipt["replicates"] || error("HMC comparator group missing")
        @printf(io, "| %d | %.2f μs | %.2f μs | %.2f μs |\n",
                steps, median(native), median(reactant), minimum(ahmc))
    end
    Markdown.parse(String(take!(io)))
end

function render_endpoint_results()
    receipt = TOML.parsefile(joinpath(DIR,"matched-endpoint-scaling-v1.toml"))
    data = read(joinpath(DIR,"matched-endpoint-scaling-v1.csv"),String)
    bytes2hex(sha256(data)) == receipt["data_sha256"] || error("Endpoint data changed")
    lines = split(chomp(data),'\n')
    headers = split(first(lines),',')
    rows = [Dict(zip(headers,split(line,','))) for line in lines[2:end]]
    io = IOBuffer()
    println(io,"| Steps | RK native | AdvancedHMC endpoint | RK Reactant | ProbProg endpoint |")
    println(io,"| --- | ---: | ---: | ---: | ---: |")
    for steps in (4,16)
        values = map(("rk_native_endpoint","ahmc_endpoint","rk_reactant_endpoint","probprog_endpoint")) do kernel
            samples = [parse(Float64,row["seconds"])*1e6/10000 for row in rows
                if row["phase"]=="execute" && row["transitions"]=="10000" &&
                   row["steps_per_transition"]==string(steps) && row["kernel"]==kernel]
            length(samples)==receipt["replicates"] || error("Endpoint samples missing")
            median(samples)
        end
        @printf(io,"| %d | %.2f μs | %.2f μs | %.2f μs | %.2f μs |\n",steps,values...)
    end
    Markdown.parse(String(take!(io)))
end
end
