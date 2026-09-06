include(joinpath(@__DIR__,"multinomial_scaling.jl"))
include(joinpath(@__DIR__,"position_multinomial_hmc_kernel.jl"))

function position_multinomial_scaling(path;cpu_loop_bytes=65536,kwargs...)
    multinomial_scaling(path;
        reactant_compile_options=TracedSlotCompiler.cpu_compile_options(
            ;small_loop_bytes=cpu_loop_bytes),
        prototype_options=(source=PositionMultinomialHMCAuthoring.multinomial_hmc_state,
            source_controls=(step_f=F.leapfrog!,stepsize=0.03)),kwargs...)
end

if abspath(PROGRAM_FILE)==@__FILE__
    1 <= length(ARGS) <= 2 || error(
        "usage: position_multinomial_scaling.jl /absolute/output.csv [CPU-loop-bytes|default]")
    limit=length(ARGS)==1 ? 65536 : ARGS[2]=="default" ? nothing : parse(Int,ARGS[2])
    position_multinomial_scaling(ARGS[1];cpu_loop_bytes=limit)
end
