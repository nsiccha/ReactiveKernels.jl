module ScanPlateADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel subject_scans(q::Vector{Float64}, X::Matrix{Float64}) = begin
    cells = plate(eachcol(X)) do xs
        seed = (value=q[1],)
        history = scan(xs; init=seed) do carry, x
            next = carry.value + q[2]*x
            ((value=next,), next)
        end
        sum(history)
    end
    total = sum(cells)
    return total
end

@testset "nested plate scans preserve ordinary reverse AD" begin
    q = [0.3, 0.7]
    for (n, G) in ((0, 2), (1, 2), (17, 7)), bound in (false, true)
        X = reshape(sin.(1:n*G), n, G)
        weight = sum((n-i+1)*X[i,s] for s in 1:G for i in 1:n; init=0.0)
        expected = n*G*q[1] + weight*q[2]
        k = bound ? prepare(subject_scans; bound=(; X)) : prepare(subject_scans)
        args = bound ? (q,) : (q, X)
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), args...; active=:q)
        @test k(args...) ≈ expected
        @test ad_gradient(ad, args...) ≈ [n*G, weight]
    end
end

# Each scan result below is summed by Base `sum`; an empty sequence must
# differentiate under ordinary Reverse like a nonempty one (see
# `benchmark/repro_enzyme_branch_allocation_phi.jl`).
@kernel summed(xs, gain) = begin
    history = scan(xs; init=0.0) do carry, x
        next = carry + x*gain
        (next, next)
    end
    total = sum(history)
    return total
end
@kernel trajectory(xs, gain) = begin
    path = scan(xs; init=gain, include_init=true) do carry, x
        next = carry + x*gain
        (next, next)
    end
    total = sum(path)
    return total
end
# The plate fuses into the scan loop; the second reader keeps its pointwise
# vector materialized.
@kernel fused(xs, gain) = begin
    history = scan(xs; init=0.0) do carry, x
        next = carry + x*gain
        (next, next)
    end
    pointwise = plate(history) do h
        h*gain
    end
    total = sum(pointwise)
    squares = sum(abs2, pointwise)
    objective = total + squares
    return objective
end

@testset "empty scans keep ordinary reverse AD" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    gain = 0.7
    for n in (0, 1, 5)
        xs = sin.(1:n)
        original = copy(xs)
        prefix = cumsum(xs)
        cases = (
            (summed, :total, gain*sum(prefix), sum(prefix)),
            (trajectory, :total, gain*(1 + sum(prefix .+ 1)), 1 + sum(prefix .+ 1)),
            (fused, :objective, gain^2*sum(prefix) + gain^4*sum(abs2, prefix),
             2gain*sum(prefix) + 4gain^3*sum(abs2, prefix)))
        for (spec, want, value, derivative) in cases
            k = prepare(spec; want)
            ad = prepare_ad(k, backend, xs, gain; active=:gain)
            got, gradient = ad_value_and_gradient(ad, xs, gain)
            @test k(xs, gain) ≈ value atol=1e-12
            @test got ≈ value atol=1e-12
            @test gradient ≈ derivative atol=1e-12
        end
        # The runtime scan op that tensorized bodies call. The sequence is a
        # `Const` argument, as `prepare_ad` passes data; a closure capturing it
        # is the separate Enzyme boundary pinned in the next testset.
        for (spec, includes_seed) in ((summed, false), (trajectory, true))
            op = only(r.op for r in spec.graph.recipes
                      if r.op isa ReactiveKernels._AuthoredScanOp)
            seed = includes_seed ? gain : 0.0
            expected = includes_seed ? vcat(gain, gain .* (prefix .+ 1)) : gain .* prefix
            derivative = includes_seed ? 1 + sum(prefix .+ 1) : sum(prefix)
            @test op(seed, xs, gain) ≈ expected
            runtime = Enzyme.autodiff(Enzyme.Reverse,
                (g, sequence) -> sum(op(includes_seed ? g : 0.0, sequence, g)),
                Enzyme.Active, Enzyme.Active(gain), Enzyme.Const(xs))
            @test first(only(runtime)) ≈ derivative atol=1e-12
        end
        @test xs == original
    end
end

# Plain Enzyme over a closure that captures its data must first prove the
# closure is never written. On Julia 1.12 and 1.13 that proof follows each
# Float64 read from the captured sequence and calls its store into the scan's
# result a capture (`EnzymeMutabilityException`). Loops without RK, such as
# `xs .* g`, fail the same way, and one shape returns 0.0 for the primal and
# the derivative without an error:
# `benchmark/repro_enzyme_closure_float_store_readonly.jl`. `prepare_ad`
# passes data as `Constant` contexts. `Const(f)` and a `Const` data argument
# differentiate the same call exactly.
@testset "plain Enzyme over a closure capturing the sequence" begin
    gain = 0.7
    xs = sin.(1:5)
    original = copy(xs)
    derivative = sum(cumsum(xs))
    k = prepare(summed; want=:total)
    captured = g -> k(xs, g)
    reverse_gradient(f) = only(only(Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
                                                    Enzyme.Active(gain))))
    @test reverse_gradient(Enzyme.Const(captured)) ≈ derivative atol=1e-12
    argument = Enzyme.autodiff(Enzyme.Reverse, (g, sequence) -> k(sequence, g),
                               Enzyme.Active, Enzyme.Active(gain), Enzyme.Const(xs))
    @test first(only(argument)) ≈ derivative atol=1e-12
    if VERSION >= v"1.12"
        # Upstream Enzyme readonly-proof gap; drop with docs/src/constraints.md.
        @test_broken reverse_gradient(captured) ≈ derivative atol=1e-12
    else
        @test reverse_gradient(captured) ≈ derivative atol=1e-12
    end
    @test xs == original
end

# A declared scan output converts every assignment to its type, so the
# lowering must never bind it to a placeholder when the step's inferred output
# type is not concrete (here `Union{Float64, Int}`).
@kernel declared_unstable(xs, gain) = begin
    history::Vector{Float64} = scan(xs; init=0.0) do carry, x
        next = x > 0 ? carry + x*gain : 0
        (next, next)
    end
    total = sum(history)
    return total
end

@testset "declared scan outputs accept a non-concrete step type" begin
    k = prepare(declared_unstable; want=:total)
    for (xs, expected) in ((Float64[], 0.0), ([0.5, -1.0, 2.0], 0.35 + 0.0 + 1.4))
        original = copy(xs)
        @test k(xs, 0.7) ≈ expected
        @test xs == original
    end
end

# A scan step that owns a child plate or scan is typed from the step's leaf
# operations (`_scan_type_slots!`). Querying the scan operation itself instead
# carried its whole step PreparedKernel, with graph and `Expr` metadata, into
# the differentiated call (`EnzymeMutabilityException` under ordinary Reverse
# from a plate cell), and typed an empty sequence's result `Any`.
@kernel subject_update(xs, w) = begin
    trajectory = scan(xs; init=zeros(length(w))) do previous, x
        next = plate(previous, w) do p, wi
            p + x*wi
        end
        (next, sum(next))
    end
    total()::Float64 = sum(trajectory)
end
@kernel endpoint_cells(groups::Vector{Vector{Float64}}, w::Vector{Float64}) = begin
    totals = plate(groups) do xs
        subject_update(xs, w).total()
    end
    t::Float64 = sum(totals)
    return t
end
@kernel scan_cells(groups::Vector{Vector{Float64}}, w::Vector{Float64}) = begin
    totals = plate(groups) do xs
        trajectory = scan(xs; init=0.0) do previous, x
            partial = scan(w; init=previous) do acc, wi
                next_acc = acc + x*wi
                (next_acc, next_acc)
            end
            next = partial[end]
            (next, next)
        end
        sum(trajectory)
    end
    t::Float64 = sum(totals)
    return t
end
@kernel nested_cells(groups::Vector{Vector{Float64}}, w::Vector{Float64}) = begin
    totals = plate(groups) do xs
        inner = plate(xs) do x
            path = scan(w; init=0.0) do a, wi
                cells = plate(w) do v
                    v*x
                end
                n = a + wi*sum(cells)
                (n, n)
            end
            sum(path)
        end
        sum(inner)
    end
    t::Float64 = sum(totals)
    return t
end
@kernel stepped(xs::Vector{Float64}, w::Vector{Float64}) = begin
    trajectory = scan(xs; init=zeros(length(w))) do previous, x
        next = plate(previous, w) do p, wi
            p + x*wi
        end
        (next, sum(next))
    end
    total::Float64 = sum(trajectory)
    return total
end

_retains_body_op(k) = any(op -> op isa Union{ReactiveKernels._AuthoredScanOp,
                                               ReactiveKernels._AuthoredPlateOp},
                          ReactiveKernels._ad_native_ops(k))
_loops(ex) = ex isa Expr ? Int(ex.head === :for) + sum(_loops, ex.args; init=0) : 0

@testset "scans whose step owns a child plate or scan keep ordinary reverse AD" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    w = [1.0, 2.0, 0.5]
    m, total_w = length(w), sum(w)
    weighted = sum((m - j + 1)*w[j] for j in 1:m)
    for groups in ([[0.3, -0.2, 0.5], Float64[], [0.1], [0.4, 0.2]],
                   [Float64[]], Vector{Float64}[])
        original = deepcopy(groups)
        prefix = sum(xs -> sum(cumsum(xs); init=0.0), groups; init=0.0)
        mass = sum(xs -> sum(xs; init=0.0), groups; init=0.0)
        cases = (
            (endpoint_cells, total_w*prefix, fill(prefix, m)),
            (scan_cells, total_w*prefix, fill(prefix, m)),
            (nested_cells, mass*total_w*weighted,
             [mass*(weighted + total_w*(m - j + 1)) for j in 1:m]))
        for (spec, value, derivative) in cases, bound in (false, true)
            k = bound ? prepare(spec; bound=(; groups)) : prepare(spec)
            args = bound ? (w,) : (groups, w)
            @test !_retains_body_op(k)
            ad = prepare_ad(k, backend, args...; active=:w)
            got, gradient = ad_value_and_gradient(ad, args...)
            @test k(args...) ≈ value atol=1e-12
            @test got ≈ value atol=1e-12
            @test gradient ≈ derivative atol=1e-12
        end
        @test isequal(groups, original)
    end
    # Bound data does not change the emitted iteration structure.
    for spec in (endpoint_cells, scan_cells, nested_cells)
        small = code_expr(prepare(spec; bound=(; groups=[[0.3]])))
        large = code_expr(prepare(spec; bound=(; groups=[sin.(1:n) for n in 1:9])))
        @test _loops(small) == _loops(large)
    end
    # The top-level scan's result is typed before any step runs.
    k = prepare(stepped)
    reader = prepare(stepped; want=:trajectory)
    @test !_retains_body_op(k)
    for xs in (Float64[], [0.3], [0.3, -0.2, 0.5])
        original = copy(xs)
        prefix = sum(cumsum(xs); init=0.0)
        ad = prepare_ad(k, backend, xs, w; active=:w)
        got, gradient = ad_value_and_gradient(ad, xs, w)
        @test eltype(reader(xs, w)) === Float64
        @test k(xs, w) ≈ total_w*prefix atol=1e-12
        @test got ≈ total_w*prefix atol=1e-12
        @test gradient ≈ fill(prefix, m) atol=1e-12
        @test xs == original
    end
end

# The inner scan reads the same, possibly empty, sequence at every outer step,
# and its loop is nested in the outer scan's loop in one generated body.
@kernel seeded_inner(xs::Vector{Float64}, us::Vector{Vector{Float64}},
                     w::Vector{Float64}) = begin
    trajectory = scan(xs; init=zeros(2)) do previous, x
        pending = scan(eachindex(us); init=previous .* w[1],
                       include_init=true) do state, i
            n = state .+ x .* us[i]
            (n, n)
        end
        next = pending[end]
        (next, sum(next))
    end
    total::Float64 = sum(trajectory) + sum(w)
    return total
end
@kernel summed_inner(xs::Vector{Float64}, us::Vector{Vector{Float64}},
                     w::Vector{Float64}) = begin
    trajectory = scan(xs; init=zeros(2)) do previous, x
        seed = previous .* w[1]
        partial = scan(eachindex(us); init=seed) do state, i
            n = state .+ x .* us[i]
            (n, sum(n))
        end
        next = seed .+ sum(partial; init=0.0)
        (next, sum(next))
    end
    total::Float64 = sum(trajectory) + sum(w)
    return total
end

# Both carries' entry sums follow σ = α*w1*σ + β*x from σ = 0; the total adds
# every σ. Returns the total and its derivative with respect to w1.
function _carry_recurrence(xs, w1, α, β)
    σ = dσ = total = dtotal = 0.0
    for x in xs
        σ, dσ = α*w1*σ + β*x, α*σ + α*w1*dσ
        total += σ
        dtotal += dσ
    end
    total, dtotal
end

@testset "scans nested over an outer-invariant empty sequence keep ordinary reverse AD" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    w = [1.3, 2.0, 0.5]
    for us in (Vector{Float64}[], [[1.0, 2.0]], [[1.0, 2.0], [0.5, -1.0], [0.0, 3.0]]),
            xs in (Float64[], [0.3], [0.3, -0.2, 0.5])
        original = (deepcopy(us), copy(xs))
        m = length(us)
        mass = sum(sum, us; init=0.0)
        prefixes = sum(i -> sum(sum, us[1:i]), 1:m; init=0.0)
        for (spec, α, β) in ((seeded_inner, 1, mass),
                             (summed_inner, 1 + 2m, 2prefixes)), bound in (false, true)
            total, dtotal = _carry_recurrence(xs, w[1], α, β)
            value, derivative = total + sum(w), [dtotal + 1, 1.0, 1.0]
            k = bound ? prepare(spec; bound=(; us)) : prepare(spec)
            args = bound ? (xs, w) : (xs, us, w)
            ad = prepare_ad(k, backend, args...; active=:w)
            got, gradient = ad_value_and_gradient(ad, args...)
            @test k(args...) ≈ value atol=1e-12
            @test got ≈ value atol=1e-12
            @test gradient ≈ derivative atol=1e-12
        end
        @test isequal((us, xs), original)
    end
end
end
