module NativeConcatADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# Constant data concatenated with parameter-dependent arrays.  Base's dense
# concatenation methods fail native Reverse static activity analysis on these
# (`benchmark/repro_enzyme_mixed_activity_concat.jl`); the native body's
# companions keep each operand's copy separate.
@kernel design(a, x, ones_col, w) = begin
    v = exp.(a .* x)
    X = hcat(ones_col, x, v)
    total = sum(X * w)
    return total
end
@kernel design_bracket(a, x, w) = begin
    X = [ones(eltype(x), length(x)) x exp.(a .* x)]
    total = sum(X * w)
    return total
end
@kernel design_matrix(a, D, x, w) = begin
    X = hcat(D, exp.(a .* x))
    total = sum(X * w)
    return total
end
@kernel stacked(a, x, r) = begin
    total = sum(vcat(x, exp.(a .* x)) .* r)
    return total
end
@kernel stacked_bracket(a, x, r) = begin
    total = sum([x; exp.(a .* x)] .* r)
    return total
end
@kernel stacked_matrix(a, C, R) = begin
    total = sum(vcat(C, a .* C) .* R)
    return total
end
@kernel blocks(a, C, x, R) = begin
    v = exp.(a .* x)
    total = sum([C v; C v] .* R)
    return total
end
@kernel cells(a, x, c) = begin
    pointwise = plate(x, a) do xi, s
        sum(vcat(c, [exp(s * xi)]))
    end
    total = sum(pointwise)
end
@kernel scanned(a, x, c) = begin
    trajectory = scan(x; init = zero(a)) do carry, xi
        next = carry + sum(vcat(c, [exp(a * xi)]))
        (next, next)
    end
    total = sum(trajectory)
end
# Scalar and mixed bracket literals of constant and active entries.
@kernel literal_cells(a, x, c) = begin
    pointwise = plate(x) do xi
        system = [-a 0; a -c]
        affine = [system [exp(a * xi), 0]; 0 0 1]
        sum(affine * [1, 2, 3])
    end
    total = sum(pointwise)
end
@kernel literal_scanned(a, x, c) = begin
    trajectory = scan(x; init = zero(a)) do carry, xi
        next = first([carry 1; 0 1] * [exp(a * xi), c])
        (next, next)
    end
    total = sum(trajectory)
end
@traceable with_column(D, v) = hcat(D, v)
@kernel helper(a, D, x, w) = begin
    total = sum(with_column(D, exp.(a .* x)) * w)
    return total
end

# Plain Julia references with fresh outputs and explicit loops.
function design_ref(a, x, w)
    sum(w[1] + w[2] * x[i] + w[3] * exp(a * x[i]) for i in eachindex(x); init = zero(a))
end
function stacked_ref(a, x, r)
    n = length(x)
    sum(x[i] * r[i] + exp(a * x[i]) * r[n + i] for i in 1:n; init = zero(a))
end
function blocks_ref(a, C, x, R)
    n = length(x)
    total = zero(a)
    for half in (0, n), i in 1:n
        total += C[i, 1] * R[half + i, 1] + C[i, 2] * R[half + i, 2] +
                 exp(a * x[i]) * R[half + i, 3]
    end
    total
end
function literal_scanned_ref(a, x, c)
    carry, total = zero(a), zero(a)
    for xi in x
        carry = carry * exp(a * xi) + c
        total += carry
    end
    total
end
_derivative(f, a) = (f(a + cbrt(eps(a))) - f(a - cbrt(eps(a)))) / (2cbrt(eps(a)))

@testset "native Reverse through concatenations of constant and active arrays" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for T in (Float64, Float32), n in (0, 1, 5)
        x = T[0.4 * sin(i) for i in 1:n]
        w = T[0.3, -0.2, 0.7]
        r = T[cos(i) for i in 1:2n]
        D = hcat(ones(T, n), x)
        C = T[0.1i + 0.2j for i in 1:n, j in 1:2]
        Cr = T[sin(i + j) for i in 1:n, j in 1:2]
        R = T[cos(i + 2j) for i in 1:2n, j in 1:2]
        Rb = T[cos(i + j) for i in 1:2n, j in 1:3]
        saved = deepcopy((x, w, r, D, C, Cr, R, Rb))
        cases = (
            (design, (; x, ones_col = ones(T, n), w), a -> design_ref(a, x, w)),
            (design_bracket, (; x, w), a -> design_ref(a, x, w)),
            (design_matrix, (; D, x, w), a -> design_ref(a, x, w)),
            (helper, (; D, x, w), a -> design_ref(a, x, w)),
            (stacked, (; x, r), a -> stacked_ref(a, x, r)),
            (stacked_bracket, (; x, r), a -> stacked_ref(a, x, r)),
            (stacked_matrix, (; C = Cr, R),
             a -> sum(Cr .* R[1:n, :]) + a * sum(Cr .* R[n+1:end, :])),
            (blocks, (; C, x, R = Rb), a -> blocks_ref(a, C, x, Rb)),
            (cells, (; x, c = w), a -> n * sum(w) + sum(exp.(a .* x); init = zero(a))),
            (scanned, (; x, c = w),
             a -> sum((n - k + 1) * (sum(w) + exp(a * x[k])) for k in 1:n; init = zero(a))),
            (literal_cells, (; x, c = T(0.25)),
             a -> sum(3exp(a * xi) - 2 * T(0.25) + 3 for xi in x; init = zero(a))),
            (literal_scanned, (; x, c = T(0.25)), a -> literal_scanned_ref(a, x, T(0.25))),
        )
        for (spec, data, reference) in cases, a in T.((0.3, -0.6))
            k = prepare(spec; bound = data)
            ad = prepare_ad(k, backend, a; active = :a)
            value, gradient = ad_value_and_gradient(ad, a)
            @test k(a) ≈ reference(a)
            @test value ≈ reference(a)
            @test gradient ≈ _derivative(reference, Float64(a)) rtol = sqrt(eps(T)) atol = sqrt(eps(T))
            @test typeof(gradient) == T
        end
        # Every operand active at once, through the unbound kernel.
        if n > 0
            k = prepare(design)
            ad = prepare_ad(k, backend, T(0.3), x, ones(T, n), w;
                            active = (:a, :x, :w))
            _, (ga, gx, gw) = ad_value_and_gradient(ad, T(0.3), x, ones(T, n), w)
            @test ga ≈ _derivative(a -> design_ref(a, x, w), 0.3) rtol = sqrt(eps(T))
            @test gx ≈ [w[2] + w[3] * T(0.3) * exp(T(0.3) * xi) for xi in x]
            @test gw ≈ [n, sum(x), sum(exp.(T(0.3) .* x))]
        end
        @test (x, w, r, D, C, Cr, R, Rb) == saved
    end
end
end
