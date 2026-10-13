# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse derivatives run slower in a Julia process started with
# more than one thread, whether or not they allocate. Enzyme's native codegen
# sets `parallel = Base.Threads.nthreads() > 1` and passes it as the
# `atomicAdd` argument of the reverse-mode transform, so in any multi-threaded
# process every shadow accumulation (`dx[i] += ...`) becomes an
# `atomicrmw fadd`. On x86-64 that is a compare-and-swap loop that also stops
# the loop from vectorizing. The choice is made at compile time from the
# process's thread count, not from whether the differentiated code runs in
# parallel, and `autodiff` exposes no option to turn it off.
#
# Run the same file twice and compare:
#     JULIA_NUM_THREADS=1 julia --project=<env with Enzyme> repro_enzyme_thread_atomic_add.jl
#     JULIA_NUM_THREADS=8 julia --project=<env with Enzyme> repro_enzyme_thread_atomic_add.jl
#
# Recorded on gordito (AMD Ryzen 9 9950X), Julia 1.10.12 with
# `-C znver3,+avx512f,+avx512dq,+avx512cd,+avx512bw,+avx512vl`, Enzyme 0.13.216
# and 0.13.206, 2026-10-13 (best time per gradient, bytes per gradient, atomic
# updates in the derivative module):
#     threads   sum-of-squares loop (n = 1000)   Gaussian log likelihood (434 x 4)
#     1         50-82 ns, 0 B, 0 atomicrmw       1213-1527 ns, 14592 B, 0 atomicrmw
#     8         3576-3608 ns, 0 B, 5 atomicrmw   4335-4476 ns, 14592 B, 14 atomicrmw
# The loop allocates nothing at either thread count and still runs about 40x
# slower; the likelihood allocates the same bytes at both. The cost therefore
# comes from the atomic shadow updates, not from allocation. ReactiveKernels'
# prepared gradients (`prepare_ad`, RKPPL `prepare_sampler`) inherit it, and
# so does a non-allocating kernel.
using Enzyme

sum_of_squares(x) = (s = 0.0; @inbounds for i in eachindex(x); s += x[i]^2; end; s)
loglik(beta, X, y) = (r = y .- X * beta; -0.5 * sum(abs2, r))

function best_ns(f; reps = 200, inner = 50)
    f()
    best = Inf
    for _ in 1:reps
        t = time_ns()
        for _ in 1:inner
            f()
        end
        best = min(best, (time_ns() - t) / inner)
    end
    round(Int, best)
end

function gradient_row(label, f, x, constants...)
    dx = zero(x)
    g() = (fill!(dx, 0.0);
           autodiff(Reverse, Const(f), Active, Duplicated(x, dx), map(Const, constants)...))
    g()
    bytes = (g(); @allocated g())
    ir = sprint(io -> Enzyme.Compiler.enzyme_code_llvm(io, f, Active,
        Tuple{Duplicated{typeof(x)},map(c -> Const{typeof(c)}, constants)...};
        dump_module = true))
    println(rpad(label, 34), lpad(best_ns(g), 7), " ns   ", lpad(bytes, 6), " B   ",
        count("atomicrmw", ir), " atomicrmw in the derivative module")
end

println(VERSION, ", Enzyme ", pkgversion(Enzyme), ", threads ", Threads.nthreads())
gradient_row("sum-of-squares loop (n = 1000)", sum_of_squares, sin.(1:1000))
X = cos.((1:434) .* (1:4)')
gradient_row("Gaussian log likelihood (434 x 4)", loglik, sin.(1:4), X, sin.(1:434))
