module RepeatedPDExample
using ReactiveKernelsPPL

# A simple analytic PD time course. Each input row is a measurement, so
# repeated subject/assay/time triples remain separate likelihood terms.
pd_mean(time, baseline, effect, rate) =
    baseline + effect * (-expm1(-rate * time))

const model = @rkppl begin
    baseline[levels(subject)] .~ Normal.(0, 1)
    effect[1:3] .~ Normal.(0, 1)
    rate ~ Exponential(1)
    mu = pd_mean.(time, baseline[subject], effect[assay], rate)
    y .~ Normal.(mu, 0.7)
end

function synthetic_data(groups=1)
    # Subject 1 has two assay-2 measurements at time zero. Input order
    # interleaves assays and retains the repeated observations.
    subject = [1, 1, 1, 1, 1, 1, 2, 3, 3, 3]
    assay = [1, 2, 2, 1, 3, 3, 2, 1, 3, 3]
    time = [12.0, 0.0, 0.0, 6.0, 2.0, 14.0, 0.0, 0.0, 4.0, 9.0]
    (; subject=vcat((subject .+ 3g for g in 0:groups-1)...),
        assay=repeat(assay, groups), time=repeat(time, groups),
        y=repeat([0.2, -0.1, 0.3, 0.4, 0.5, 0.1, -0.2, 0.2, 0.6, 0.4], groups))
end
end
