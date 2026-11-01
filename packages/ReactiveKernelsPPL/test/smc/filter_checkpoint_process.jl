using ReactiveKernels, ReactiveKernelsPPL, Random, Distributions, SequentialMonteCarlo, Serialization

# A fresh generated module each time exercises portable source identity too.
function target()
    mod=Module(gensym(:FilterCheckpoint))
    Core.eval(mod,:(using ReactiveKernelsPPL, Distributions))
    model=Core.eval(mod,quote
        @rkppl begin
            phi ~ Normal()
            @scan begin
                h[1] ~ Normal()
                for t in 2:T
                    h[t] ~ Normal(phi*h[t-1],0.3)
                end
            end
            y .~ Normal.(h,0.5)
        end
    end)
    prepare_filter(model;model_id="fresh-process-filter",parameters=(;phi=0.8))
end
snapshot(s)=(;s.carries,s.weights,s.parameter_mass,s.time,s.predictive_logscore,s.log_normalizer,
    s.ancestors,s.lineage,s.group_times,s.history,rng=rand(copy(s.rng),UInt64,8))
advance(s)=filter_stream(s,((;observation=(;y=0.3sin(t))) for t in 1:7))
if ARGS[1]=="write"
    s=initialize_filter(target();rng=Xoshiro(42),particles=128,
        ensemble=[(;phi=0.5),(;phi=0.8)],weights=[0.2,0.8])
    s=advance(s)
    serialize(ARGS[2],(;checkpoint=filter_checkpoint(s),expected=snapshot(advance(s))))
else
    saved=deserialize(ARGS[2])
    s=resume_filter(target(),saved.checkpoint)
    @assert snapshot(advance(s))==saved.expected
    println("Fresh-process filtering continuation passed")
end
