using Serialization
include("prepared_fixture.jl")
using .PreparedSMCFixture, ReactiveKernelsPPL, Random
const F = PreparedSMCFixture

old = F.target(F.GaussianFactory(; scale=[0.9,1.1]); binding_id="old")
new = F.target()
if ARGS[1] == "write"
    schedule = tune_smc(new; rng=Xoshiro(800), init=zeros(2), pilot_steps=8,
        temperatures=(0.0,1.0), options=F.options())
    initial = smc(new; rng=Xoshiro(71), particles=97, stages=1,
        options=SMCOptions(tuning=schedule, nthreads=1))
    update = update_smc(smc(old; rng=Xoshiro(72),particles=97,options=F.options()),
        new; stages=1)
    serialize(ARGS[2], (; initial=smc_checkpoint(initial), update=smc_checkpoint(update),
        initial_result=F.snapshot(smc(initial)), update_result=F.snapshot(smc(update))))
elseif ARGS[1] == "read"
    saved = deserialize(ARGS[2])
    @assert F.snapshot(smc(resume_smc(new,saved.initial))) == saved.initial_result
    @assert F.snapshot(smc(resume_smc(new,saved.update;previous=old))) == saved.update_result
    println("Prepared-density fresh-process tuned batch/update continuation passed")
else
    error("expected write or read")
end
