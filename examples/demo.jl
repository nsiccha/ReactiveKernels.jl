# Runnable walkthrough of ReactiveKernels.jl.
#   julia --project=. examples/demo.jl
using ReactiveKernels

hr(t) = println("\n", "="^70, "\n", t, "\n", "="^70)

# ---------------------------------------------------------------------------
hr("1. Alternative-producer planning (gist §21 acceptance example)")

g = Graph()
u = value!(g, :u, Float64)
a = value!(g, :a, Float64)
b = value!(g, :b, Float64)
c = value!(g, :c, Float64)
r = value!(g, :r, Float64)

cheap_a(u)     = u + 0.0
combined_ab(u) = (u + 1.0, u + 10.0)   # produces a AND b
make_b(a)      = a + 100.0
make_c(a)      = a + 1000.0
finish(b, c)   = b + c

add!(g, u => a,      cheap_a;     cost = 1.0)
add!(g, u => (a, b), combined_ab; cost = 1.2)
add!(g, a => b,      make_b;      cost = 1.0)
add!(g, a => c,      make_c;      cost = 1.0)
add!(g, (b, c) => r, finish;      cost = 1.0)

println("Query A — want `a` only:")
println(explain(plan(g; have = (u,), want = (a,))))

println("\nQuery B — want `r` (combined_ab wins because b is needed too):")
pr = plan(g; have = (u,), want = (r,))
println(explain(pr))

println("\nQuery C — have (a, b) already; only make_c + finish remain:")
pc = plan(g; have = (a, b), want = (r,))
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

g2 = Graph()
X = value!(g2, :X, Float64)
location = value!(g2, :location, Float64)
scale    = value!(g2, :scale, Float64)
standardized = value!(g2, :standardized, Float64)

runs = Ref(0)
add!(g2, X => location, x -> (runs[] += 1; x))            # "mean"
add!(g2, X => scale,    x -> (runs[] += 1; abs(x) + 1.0)) # "std"
add!(g2, (X, location, scale) => standardized, (x, l, s) -> (x - l) / s)

# Phase 1: derive stats from training data.
st = ReactiveState(g2; materialize = (location, scale))
set!(st, X, 4.0)
loc, sc = get!(st, (location, scale))
println("phase-1 stats: location=$loc scale=$sc  (producer runs so far: $(runs[]))")
cp = checkpoint(st, (location, scale))

# Phase 2: new data, frozen phase-1 stats — their producers must NOT re-run.
st2 = ReactiveState(g2; frozen = cp)
set!(st2, X, 40.0)
y = get!(st2, standardized)
println("phase-2 standardized(X=40) = ", y, "   (producer runs: $(runs[]) — unchanged)")

println("\nAll demos completed.")
