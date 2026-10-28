# Standalone CPU reproducer: only Reactant and Enzyme are required.
# The PK recurrence's shape: a loop over schedule rows whose body branches on
# the row kind, with a second branch (the dose count) inside one of those
# branches. The primal is exact; the compiled gradient was silently wrong
# (-11.208 instead of -17.273) because the reverse of the inner if read its
# result adjoint without zeroing it, so a later row consumed an earlier row's
# adjoint again. Compiling it at all needs the fix that
# repro_reactant_while_reverse.jl reproduces.
using Reactant, Enzyme

const KIND = [3, 1, 1, 1, 2, 1]                 # 1: read, otherwise dose
const DT = [0.0, 24.0, 24.0, 0.0, 0.0, 5.0]
const AMOUNT = [100.0, 0.0, 0.0, 0.0, 50.0, 0.0]
const COUNT = [4, 0, 0, 0, 1, 0]

function branch(pred, yes, no, args)
    Reactant.@trace if pred
        result = yes(args...)
    else
        result = no(args...)
    end
    result
end

observe(x, acc, a, amount, count) = (x, acc + x)
dose(x, acc, a, amount, count) =
    (branch(count > 1, (y, a) -> y * a, (y, a) -> y, (x + amount, a)), acc)

function row(carry, a, kind, dt, amount, count)
    x = carry[1] * exp(-a * dt)
    branch(kind == 1, observe, dose, (x, carry[2], a, amount, count))
end

function recurrence(q)
    a = exp(Reactant.@allowscalar q[1])
    kind, dt, amount, count = map(c -> Reactant.promote_to(Reactant.TracedRArray, c),
        (KIND, DT, AMOUNT, COUNT))
    carry = (zero(a), zero(a))
    Reactant.@trace for i in 1:length(KIND)
        carry = row(carry, a, Reactant.@allowscalar(kind[i]), Reactant.@allowscalar(dt[i]),
            Reactant.@allowscalar(amount[i]), Reactant.@allowscalar(count[i]))
    end
    carry[2]
end

# p = log(0.1); exact values from 60-digit dual numbers.
q = [log(0.1)]
rq = Reactant.to_rarray(q)
primal = Reactant.@compile sync=true recurrence(rq)
@assert isapprox(Float64(primal(rq)), 31.4482233985753; rtol=1e-12)
gradient(q) = only(Enzyme.gradient(Enzyme.Reverse, recurrence, q))
compiled_gradient = Reactant.@compile sync=true gradient(rq)
got = only(Array(compiled_gradient(rq)))
@assert isapprox(got, -17.27341715259027; rtol=1e-12) "compiled gradient $got, exact -17.27341715259027"
println("compiled gradient matches the exact value: ", got)
