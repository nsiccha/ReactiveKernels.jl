# Run in a consumer environment containing RK, RKPPL, BRM and StanBlocks.
# See README.md for the source pin and invocation. No sampling is performed.
using ReactiveKernels, ReactiveKernelsPPL, BayesianRegressionModels, StanBlocks

const BRM = BayesianRegressionModels
const PPL = ReactiveKernelsPPL
const source_root = isempty(ARGS) ? pkgdir(BRM) : abspath(ARGS[1])
const output_root = length(ARGS) < 2 ? mktempdir(; prefix="brm-docs-retained-", cleanup=false) : abspath(ARGS[2])
mkpath(output_root)

module Mesh end
module Station end
module OU end
module Wastewater end

function load_model(name)
    if name == "mesh"
        path = joinpath(source_root, "docs/src/space-time-fields.md")
        blocks = [m.captures[1] for m in eachmatch(r"raw\"\"\"(.*?)\"\"\""s, read(path, String))]
        # Evaluate the original documentation block verbatim.
        source = only(filter(b -> occursin("mesh=(vertices=xy, cells=triangles)", b), blocks))
        Core.eval(Mesh, :(using BayesianRegressionModels))
        Base.include_string(Mesh, source, path)
        return Mesh.model
    elseif name == "station"
        path = joinpath(source_root, "research/space_time_fields/fields.jl")
        source = read(path, String)
        marker = "# Everything above this line also runs when Documenter displays the model."
        occursin(marker, source) || error("station source boundary not found")
        Base.include_string(Station, first(split(source, marker; limit=2)), path)
        return Base.invokelatest(Station.station_field_model)
    elseif name == "ou"
        Base.include(OU, joinpath(source_root, "research/ema_ctsem/docs_model.jl"))
        return Base.invokelatest(OU.state_space_model)
    elseif name == "wastewater"
        Base.include(Wastewater, joinpath(source_root, "research/wastewater/cdc_ww_inference.jl"))
        Base.include(Wastewater, joinpath(source_root, "research/wastewater/data.jl"))
        return Base.invokelatest(Wastewater.wastewater_public_model)
    end
    error("unknown case $name")
end

function probe(name)
    stage = "model"
    println("START ", name); flush(stdout)
    try
        model = load_model(name)
        stage = "artifact"
        artifact = Base.invokelatest(BRM.emit_rk_artifact, model; case_id=name)
        open(joinpath(output_root, name * "-artifact.txt"), "w") do io
            show(io, MIME("text/plain"), artifact.defs); println(io)
            show(io, MIME("text/plain"), artifact.ast); println(io)
        end
        stage = "translate"
        bound = Base.invokelatest(BRM.rk_translate_artifact, artifact)
        stage = "build"
        built = Base.invokelatest(PPL.build_kernel, bound)
        open(joinpath(output_root, name * "-program.txt"), "w") do io
            show(io, MIME("text/plain"), built.program)
            println(io, "\nLAYOUT ", built.layout)
        end
        stage = "jax_source"
        source = Base.invokelatest(PPL.jax_source, built, bound)
        write(joinpath(output_root, name * ".py"), source.text)
        println("PASS ", name, " dimension=", built.layout.total,
                " bytes=", ncodeunits(source.text))
        return true
    catch error
        println("FAIL ", name, " stage=", stage)
        showerror(stdout, error); println()
        for frame in Iterators.take(stacktrace(catch_backtrace()), 8)
            println(frame.func, " ", frame.file, ":", frame.line)
        end
        return false
    finally
        flush(stdout)
    end
end

println("VERSIONS Julia=", VERSION, " RK=", pkgversion(ReactiveKernels),
        " PPL=", pkgversion(PPL), " BRM=", pkgversion(BRM), " SB=", pkgversion(StanBlocks))
println("SOURCES ", source_root, "\nOUTPUT ", output_root)
cases = length(ARGS) > 2 ? ARGS[3:end] : ["mesh", "station", "ou", "wastewater"]
results = [probe(name) for name in cases]
println("RESULT ", count(identity, results), "/", length(results), " emitted JAX")
exit(all(results) ? 0 : 1)
