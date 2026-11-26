# From the repository root, use a project that develops this checkout and its PPL package:
# julia --project=packages/ReactiveKernelsPPL benchmark/jax_emitter_iteration.jl
# In the Julia REPL, starting at the repository root:
#   import Revise, ReactiveKernels, ReactiveKernelsPPL
#   include("benchmark/jax_emitter_iteration.jl")
#   built, bound = JAXEmitterIteration.fixture()
#   first_export = @time JAXEmitterIteration.emit(built, bound)
#   # Save a method edit in ReactiveKernels, then at the prompt:
#   Revise.revise(ReactiveKernels; force=false)
#   candidate = @time JAXEmitterIteration.emit(built, bound)
#   JAXEmitterIteration.check_fresh(candidate)
module JAXEmitterIteration

using ReactiveKernelsPPL, Serialization
include(joinpath(pkgdir(ReactiveKernelsPPL), "test", "jax_ppl_data.jl"))

function fixture()
    model = @rkppl begin
        theta ~ Normal(0, 1)
        y .~ Normal.(sin(theta), 1)
    end
    plan = model() | (; y = [0.25, -0.5])
    return build_kernel(plan), plan
end

function emit(built, plan)
    src = ReactiveKernelsPPL.jax_source(built, plan)
    prepare_data = only(f for f in src.functions if f.name == "prepare_data")
    names = first.(prepare_data.inputs)
    inputs = collect(zip(names, jax_raw_data(plan, src)))
    return (; source = src.text, inputs = deepcopy(inputs))
end

# Rebuild this script's fixture with the on-disk emitter in a fresh process.
# Only the emitted text and the names and values of its raw inputs cross processes.
function check_fresh(candidate)
    mktempdir() do dir
        path = joinpath(dir, "control.jls")
        project = dirname(Base.active_project())
        run(`$(Base.julia_cmd()) --startup-file=no --project=$project $(@__FILE__) --control $path`)
        control = deserialize(path)
        candidate.source == control.source || error("emitted module differs from fresh export")
        isequal(candidate.inputs, control.inputs) || error("bound inputs differ from fresh export")
    end
    println("Module and bound inputs match the fresh export.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        let (built, bound) = @time "build and bind fixture" fixture()
            first_export = @time "first emission" emit(built, bound)
            unchanged = @time "unchanged emission" emit(built, bound)
            @assert isequal(first_export, unchanged)
            check_fresh(unchanged)
        end
    elseif length(ARGS) == 2 && first(ARGS) == "--control"
        serialize(ARGS[2], emit(fixture()...))
    else
        error("usage: julia --project=packages/ReactiveKernelsPPL benchmark/jax_emitter_iteration.jl")
    end
end

end
