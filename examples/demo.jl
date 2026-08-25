# Runnable walkthrough of ReactiveKernels.jl.
#   julia --project=. examples/demo.jl
using ReactiveKernels

hr(t) = println("\n", "="^70, "\n", t, "\n", "="^70)

# ---------------------------------------------------------------------------
hr("1. Alternative-producer planning (gist §21 acceptance example)")

cheap_a(u)     = u + 0.0
combined_ab(u) = (u + 1.0, u + 10.0)   # produces a AND b
make_b(a)      = a + 100.0
make_c(a)      = a + 1000.0
finish(b, c)   = b + c

g = @kernel begin
    u::Float64
    @recipe (cost = 1.0) a::Float64 = cheap_a(u)
    @recipe (cost = 1.2) (a, b::Float64) = combined_ab(u)
    @recipe (cost = 1.0) b = make_b(a)
    @recipe (cost = 1.0) c::Float64 = make_c(a)
    @recipe (cost = 1.0) r::Float64 = finish(b, c)
    return r
end

println("Query A — want `a` only:")
println(explain(plan(g; want = :a)))

println("\nQuery B — want `r` (combined_ab wins because b is needed too):")
pr = plan(g)
println(explain(pr))

println("\nQuery C — have (a, b) already; only make_c + finish remain:")
pc = plan(g; have = (:a, :b))
println(explain(pc))
println("\nGenerated kernel for Query C:")
println(code_expr(pc))

# ---------------------------------------------------------------------------
hr("2. Hot path is ordinary, zero-allocation Julia")

kc = prepare(pc)
println("kc(2.0, 3.0) = ", kc(2.0, 3.0))          # c = make_c(2) = 1002; r = 3 + 1002 = 1005
alloc(k, x, y) = (k(x, y); @allocated k(x, y))    # measure inside a function barrier
println("allocations per call after warmup: ", alloc(kc, 2.0, 3.0))
println("\n@code_typed of the kernel call (specialized, type-stable):")
show(stdout, "text/plain", first(code_typed(kc, Tuple{Float64,Float64})))
println()

# ---------------------------------------------------------------------------
hr("3. Reactive replay with a frozen cut point (standardization)")

runs = Ref(0)
g2 = @kernel begin
    X::Float64
    location::Float64 = (runs[] += 1; X)            # "mean"
    scale::Float64 = (runs[] += 1; abs(X) + 1.0)    # "std"
    standardized::Float64 = (X - location) / scale
    return standardized
end

# Phase 1: derive stats from training data.
st = ReactiveState(g2; materialize = (:location, :scale))
set!(st, g2.X, 4.0)
loc, sc = get!(st, (g2.location, g2.scale))
println("phase-1 stats: location=$loc scale=$sc  (producer runs so far: $(runs[]))")
cp = checkpoint(st, (g2.location, g2.scale))

# Phase 2: new data, frozen phase-1 stats — their producers must NOT re-run.
st2 = ReactiveState(g2; frozen = cp)
set!(st2, g2.X, 40.0)
y = get!(st2, g2.standardized)
println("phase-2 standardized(X=40) = ", y, "   (producer runs: $(runs[]) — unchanged)")

println("\nAll demos completed.")
