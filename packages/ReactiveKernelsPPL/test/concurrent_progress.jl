# Shared by test_concurrent_build.jl and its two-thread child process.
using ReactiveKernelsPPL

# K independent Gaussian regressions on one predictor; `seed` changes only the
# data, so every K shares one construction shape.
function _ccb_wide(K::Int, seed::Int)
    body = Expr(:block)
    cols = Dict{Symbol,AbstractVector}(:x => collect(range(-1, 1; length = 20)))
    names = Symbol[:x]
    for k in 1:K
        a, b, s, y, mu = Symbol(:a, k), Symbol(:b, k), Symbol(:s, k),
                         Symbol(:y, k), Symbol(:mu, k)
        push!(body.args, :($a ~ Normal(0, 5)), :($b ~ Normal(0, 2)),
              :($s ~ Exponential(1)), :($mu = $a .+ $b .* x),
              :($y .~ Normal.($mu, $s)))
        cols[y] = sin.(cols[:x] .* (k + seed)) .+ 0.1k
        push!(names, y)
    end
    bind_data(lower_rkppl(body, Tuple(names); conditioned = Tuple(names)), cols)
end

# Wall time of two equal builds run concurrently, relative to the same two
# builds run one after the other (best of `attempts`, each with fresh data).
# About 1 when builds serialize each other: a lock held for a whole build, or
# construction dominated by compilation, which Julia serializes.
function _ccb_progress(; K::Int = 12, attempts::Int = 3)
    build_kernel(_ccb_wide(K, 0))   # compile this shape's construction path
    best = Inf
    for attempt in 1:attempts
        plans = [_ccb_wide(K, 10attempt + i) for i in 1:4]
        start = time()
        build_kernel(plans[1])
        build_kernel(plans[2])
        serial = time() - start
        start = time()
        foreach(wait, [Threads.@spawn(build_kernel(plan)) for plan in plans[3:4]])
        best = min(best, (time() - start) / serial)
        best < 0.8 && break
    end
    best
end
