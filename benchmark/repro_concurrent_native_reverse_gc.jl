# Standalone CPU reproducer: ReactiveKernels, ReactiveKernelsPPL,
# Distributions, Enzyme and DifferentiationInterface are required.
#
# A native Enzyme reverse gradient that runs while another thread of the same
# process allocates and collects garbage can use freed memory. In the runs
# below, Julia 1.10 processes mostly died with `signal (11)` (the reporter
# also saw `GC error (probable corruption)`, exit 134), and Julia 1.12 runs
# returned wrong gradients with no error. Multi-chain samplers and
# multi-threaded servers take gradients this way: one prepared sampler per
# thread. The defect is upstream: an Enzyme-only program (a `@noinline`
# function that fills a heap work vector and reads it back) crashes the same
# way on Julia 1.10 with no ReactiveKernels code, while its `NTuple` twin does
# not (Enzyme snag `julia-1-10-concu-152a4415`). One gradient thread beside
# one allocating thread is enough, so separate samplers are not racing each
# other. The same program on one thread is clean.
#
# Run (the arguments are the seconds of stress, the model, and the thread
# pool of the allocating task):
#     julia --project=<env> -t 4 repro_concurrent_native_reverse_gc.jl 20 re
#     julia --project=<env> -t 4 repro_concurrent_native_reverse_gc.jl 20 glm
#     julia --project=<env> -t 1 repro_concurrent_native_reverse_gc.jl 20 re
#     julia --project=<env> -t 1,1 repro_concurrent_native_reverse_gc.jl 20 re interactive
# `re` is a correlated random-effects model (a gather of `R = Z * F'` rows and
# an LKJ Cholesky factor); `glm` is a broadcast regression with no gather or
# matrix product. Each default-pool thread but one prepares its own sampler
# (at least one does) and compares every gradient bitwise with a
# single-threaded reference while one task allocates and calls
# `GC.gc(false)`. With one thread the tasks take turns, so nothing runs
# concurrently. The last form keeps the gradient on the only default thread
# and allocates on an interactive thread. The script exits 1 on any mismatch;
# a crash ends the process with the signal's status.
#
# Recorded on gordito (x86-64), Enzyme 0.13.216, Enzyme_jll 0.0.303,
# DifferentiationInterface 0.7.21, 2026-10-13, one 20 s run per row except
# where two are listed:
#     Julia    threads  model  allocating task  outcome
#     1.10.12  4        re     default pool     1 wrong gradient of 3,235,102
#     1.10.12  4        glm    default pool     none wrong of 26,019,776
#     1.10.12  1,1      re     interactive      segfault (exit 139)
#     1.12.7   1,1      re     interactive      none wrong of 843,583; then
#                                               1 wrong of 891,203 (value off
#                                               by 10.7, an entry by 21.3)
#     1.12.7   1        re     default pool     none wrong of 2,074,998
# The same random-effects model in the reporter's three-sampler program
# segfaulted in 2 of 2 runs on Julia 1.10.12; on Julia 1.12.7 it returned 3
# wrong gradients of 1,830,086 in one run and in the other raised a
# `DomainError` from `log` of a negative number in the LKJ log-Jacobian,
# whose argument is positive for every input. A `prepare_nonallocating`
# gradient of the same model (18,000 B per call) segfaulted too (Julia
# 1.10.12, 4 threads).
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Random
import Enzyme
using DifferentiationInterface: AutoEnzyme

seconds = parse(Float64, get(ARGS, 1, "20"))
variant = get(ARGS, 2, "re")
churn_pool = Symbol(get(ARGS, 3, "default"))
churn_pool in (:default, :interactive) ||
    error("the allocating task's pool must be `default` or `interactive`")
println("Julia ", VERSION, ", Enzyme ", pkgversion(Enzyme), ", threads ",
    Threads.nthreads(), "+", Threads.nthreads(:interactive), " interactive, model ",
    variant, ", allocating task on the ", churn_pool, " pool")
flush(stdout)

model = if variant == "re"
    @rkppl begin
        a ~ Normal(0, 5)
        b1 ~ Normal(0, 2)
        b2 ~ Normal(0, 2)
        sigma ~ Exponential(1.0)
        Z[levels(g), 1:3] .~ Normal.(0, 1)
        sd[1:3] .~ Exponential.(1)
        L ~ LKJCholesky(3, 2.0)
        F = sd .* L
        R = Z * F'
        mu = a .+ b1 .* x .+ b2 .* z .+ R[g, 1] .+ R[g, 2] .* x .+ R[g, 3] .* z
        y .~ Normal.(mu, sigma)
    end
elseif variant == "glm"
    @rkppl begin
        a ~ Normal(0, 5)
        b1 ~ Normal(0, 2)
        b2 ~ Normal(0, 2)
        sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b1 .* x .+ b2 .* z, sigma)
    end
else
    error("model must be `re` or `glm`, got $(repr(variant))")
end

rng = Xoshiro(3)
G, n = 12, 6
g = repeat(["g$i" for i in 1:G]; inner = n)
x = randn(rng, G * n)
z = randn(rng, G * n)
y = 0.3 .+ 0.5 .* x .- 0.2 .* z .+ 0.3 .* randn(rng, G * n)
plan = variant == "re" ? model(; x, z, g) | (; y) : model(; x, z) | (; y)
built = build_kernel(plan)
d = built.layout.total
backend = AutoEnzyme(; mode = Enzyme.Reverse)
sampler() = prepare_sampler(built, plan, zeros(d); backend)
function value_and_gradient(q, u)
    gradient = similar(u)
    value, _ = sampler_value_and_gradient!(q, gradient, u)
    value, gradient
end
points = [0.3 .* randn(Xoshiro(10 + i), d) for i in 1:8]
reference = let q = sampler()
    [value_and_gradient(q, p) for p in points]
end

stop = Threads.Atomic{Bool}(false)
churn = Threads.@spawn churn_pool while !stop[]
    junk = [rand(64) for _ in 1:100]
    rand() < 0.01 && GC.gc(false)
    yield()
end
count = Threads.Atomic{Int}(0)
bad = Threads.Atomic{Int}(0)
first_error = Ref{Any}(nothing)
error_lock = ReentrantLock()
function record_mismatch(value, expected)
    Threads.atomic_add!(bad, 1)
    lock(error_lock) do
        first_error[] === nothing || return
        first_error[] = (value = value[1] - expected[1],
            gradient = maximum(abs, value[2] .- expected[2]))
    end
end
workers = map(1:max(1, Threads.nthreads() - 1)) do t
    Threads.@spawn begin
        q = sampler()
        deadline = time() + seconds
        i = 0
        while time() < deadline
            i += 1
            j = mod1(i + t, length(points))
            value = value_and_gradient(q, points[j])
            Threads.atomic_add!(count, 1)
            isequal(value, reference[j]) || record_mismatch(value, reference[j])
        end
    end
end
foreach(wait, workers)
stop[] = true
wait(churn)
println("gradients=", count[], " mismatches=", bad[])
first_error[] === nothing || println("first mismatch: value off by ",
    first_error[].value, ", largest gradient entry off by ", first_error[].gradient)
exit(bad[] == 0 ? 0 : 1)
