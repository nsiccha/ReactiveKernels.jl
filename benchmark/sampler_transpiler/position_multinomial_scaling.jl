include(joinpath(@__DIR__,"multinomial_scaling.jl"))
include(joinpath(@__DIR__,"position_multinomial_hmc_kernel.jl"))

function position_multinomial_scaling(path;kwargs...)
    multinomial_scaling(path;
        prototype_options=(source=PositionMultinomialHMCAuthoring.multinomial_hmc_state,
            source_controls=(step_f=F.leapfrog!,stepsize=0.03)),kwargs...)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: position_multinomial_scaling.jl /absolute/output.csv")
    position_multinomial_scaling(only(ARGS))
end
