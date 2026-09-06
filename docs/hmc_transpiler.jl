module HMCTranspilerDocs
using Markdown, SHA, Statistics, TOML, Printf

const DIR = normpath(joinpath(@__DIR__, "..", "benchmark", "sampler_transpiler"))

render_source() = Markdown.MD(Markdown.Code("julia",
    read(joinpath(DIR, "position_multinomial_hmc_kernel.jl"), String)))

function render_results()
    receipt = TOML.parsefile(joinpath(DIR, "position-multinomial-scaling-v2.toml"))
    data = read(joinpath(DIR, "position-multinomial-scaling-v2.csv"), String)
    bytes2hex(sha256(data)) == receipt["data_sha256"] || error("HMC scaling data changed")
    lines = split(chomp(data), '\n')
    headers = split(first(lines), ',')
    rows = [Dict(zip(headers, split(line, ','))) for line in lines[2:end]]
    probprog_receipt = TOML.parsefile(joinpath(DIR, "probprog-hmc-scaling-v1.toml"))
    probprog_data = read(joinpath(DIR, "probprog-hmc-scaling-v1.csv"), String)
    bytes2hex(sha256(probprog_data)) == probprog_receipt["data_sha256"] ||
        error("ProbProg scaling data changed")
    probprog_lines = split(chomp(probprog_data), '\n')
    split(first(probprog_lines), ',') == headers || error("ProbProg scaling schema mismatch")
    probprog_rows = [Dict(zip(headers, split(line, ','))) for line in probprog_lines[2:end]]
    execute = filter(row -> row["phase"] == "execute", rows)
    length(execute) == receipt["execute_rows"] || error("HMC execution rows missing")
    for row in execute
        parse(Int, row["gradient_evaluations"]) ==
            parse(Int, row["steps_per_transition"]) * parse(Int, row["transitions"]) ||
            error("HMC integration-work axis mismatch")
    end
    io = IOBuffer()
    println(io, "| Steps | Native median | Reactant median | AdvancedHMC fastest sample | ProbProg endpoint HMC median |")
    println(io, "| --- | ---: | ---: | ---: | ---: |")
    for steps in (4, 16)
        selected = filter(row -> row["steps_per_transition"] == string(steps) &&
                                  row["transitions"] == "10000", execute)
        times(backend) = [parse(Float64, row["seconds"]) * 1e6 / 10000
                          for row in selected if row["backend"] == backend]
        native, reactant, ahmc = times("native"), times("Reactant"), times("AdvancedHMC")
        length(native) == length(reactant) == receipt["replicates"] ||
            error("HMC replicates missing")
        length(ahmc) == 2 * receipt["replicates"] || error("HMC comparator group missing")
        probprog = [parse(Float64, row["seconds"]) * 1e6 / 10000
                    for row in probprog_rows if row["phase"] == "execute" &&
                    row["steps_per_transition"] == string(steps) && row["transitions"] == "10000"]
        length(probprog) == probprog_receipt["replicates"] || error("ProbProg replicates missing")
        @printf(io, "| %d | %.2f μs | %.2f μs | %.2f μs | %.2f μs |\n",
                steps, median(native), median(reactant), minimum(ahmc), median(probprog))
    end
    Markdown.parse(String(take!(io)))
end
end
