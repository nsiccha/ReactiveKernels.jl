module HMCTranspilerDocs
using Markdown, SHA, Statistics, TOML, Printf

const DIR = normpath(joinpath(@__DIR__, "..", "benchmark", "sampler_transpiler"))

render_source(filename="position_multinomial_hmc_kernel.jl") =
    Markdown.MD(Markdown.Code("julia", read(joinpath(DIR, filename), String)))

function render_consumer(filename, marker)
    source = read(joinpath(DIR, filename), String)
    start, stop = "# BEGIN $marker\n", "# END $marker"
    count(start, source) == count(stop, source) == 1 || error("consumer source marker drift")
    body = split(split(source, start; limit=2)[2], stop; limit=2)[1]
    Markdown.MD(Markdown.Code("julia", body))
end

function run_consumers()
    # Frozen for the docs build (user decision `00jueci`): the scalar, vector and
    # HMC transpiler consumers import Enzyme for reverse-mode gradients
    # (benchmark/sampler_transpiler/eight_schools_density.jl). They are exercised
    # by the executable benchmark suite under benchmark/sampler_transpiler and its
    # tests, not during the docs build, so the build carries no Enzyme/LLVM
    # autodiff toolchain. Their verified outputs are stated here as prose instead
    # of re-executed at build time.
    Markdown.parse(
        "The executable examples produce scalar outputs **9 → 21** (state replay: " *
        "**9**) and continued vector position **[0.125, 0.25]** (derived `squared` " *
        "**[0.015625, 0.0625]**). The HMC consumer runs the same compiler with " *
        "reverse-mode gradients. These consumers are exercised by the executable " *
        "benchmark suite under " *
        "[`benchmark/sampler_transpiler`](https://github.com/nsiccha/ReactiveKernels.jl/tree/main/benchmark/sampler_transpiler), " *
        "not during the docs build.")
end

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

function render_interface_results()
    receipt = TOML.parsefile(joinpath(DIR, "prepared-interface-v2.toml"))
    data = read(joinpath(DIR, "prepared-interface-v2.csv"), String)
    bytes2hex(sha256(data)) == receipt["data_sha256"] || error("interface data changed")
    lines = split(chomp(data), '\n')
    headers = split(first(lines), ',')
    rows = [Dict(zip(headers, split(line, ','))) for line in lines[2:end]]
    count(r -> r["phase"] == "execute", rows) == receipt["execute_rows"] || error("interface rows missing")
    io = IOBuffer()
    println(io, "| Backend | Steps | Direct compiler | Prepared interface |")
    println(io, "| --- | ---: | ---: | ---: |")
    for backend in ("native", "reactant"), steps in (4, 16)
        times = map(("direct", "prepared")) do path
            samples = [parse(Float64, r["seconds"]) * 1e6 / parse(Int, r["transitions"])
                for r in rows if r["phase"] == "execute" && r["backend"] == backend &&
                    r["steps"] == string(steps) && r["path"] == path]
            length(samples) == receipt["replicates"] || error("interface samples missing")
            median(samples)
        end
        @printf(io, "| %s | %d | %.2f μs | %.2f μs |\n", backend, steps, times...)
    end
    Markdown.parse(String(take!(io)))
end
end
