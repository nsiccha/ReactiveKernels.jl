module ScalarCellReadTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel anchored(X::AbstractMatrix{Float64}, beta::AbstractVector{Float64},
                 y::AbstractVector{Float64}, key::AbstractVector{Int}) = begin
    basis = exp.(0.2 .* X)
    a = basis * beta
    weight = exp.(a[1])
    shift = a[2] .- a[1]
    mu = weight .+ shift .+ a
    dens = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

# The shared predictor has its own row axis. Its literal rows must never be
# replaced by the likelihood's selected rows, or broadcast against that axis.
@kernel separate(X::AbstractMatrix{Float64}, beta::AbstractVector{Float64},
                 eta::AbstractVector{Float64}, y::AbstractVector{Float64},
                 key::AbstractVector{Int}) = begin
    basis = exp.(0.2 .* X)
    shared = basis * beta
    weight = exp(shared[1])
    shift = shared[2] .- shared[1]
    mu = weight .+ shift .+ exp.(eta)
    dens = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init = 0)
astsize(x) = 0

@kernel indexed(x::AbstractVector{Float64}, scale::Float64,
                key::AbstractVector{Int}) = begin
    a = scale .* x
    anchor = getindex(a, 3)
    dens = plate(x) do xi
        anchor + xi
    end
    total = sum(dens)
    return total
end

# One element of a matrix: at the cell's row (`b[q, 2]`) and at a literal row
# (`b[1, 2]`). A row of `b = Z * M` reads only that row of `Z`.
@kernel matrix_read(Z::AbstractMatrix{Float64}, s::Float64, y::AbstractVector{Float64},
                    rows::AbstractVector{Int}, key::AbstractVector{Int}) = begin
    M = [1.0 s; 0.0 1.0]
    b = Z * M
    anchor = exp(b[1, 2])
    dens = plate(rows, y) do q, yi
        -(anchor * b[q, 2] - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

# A row of this matrix is computed from another row: it stays cross-row work.
@kernel matrix_read_reversed(Z::AbstractMatrix{Float64}, y::AbstractVector{Float64},
                             rows::AbstractVector{Int}, key::AbstractVector{Int}) = begin
    b = reverse(Z; dims = 1)
    dens = plate(rows, y) do q, yi
        -(b[q, 2] - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

# The same read of a matrix port, of bound data and of a computed vector.
@kernel matrix_read_port(X::AbstractMatrix{Float64}, y::AbstractVector{Float64},
                         rows::AbstractVector{Int}, key::AbstractVector{Int}) = begin
    dens = plate(rows, y) do q, yi
        -(X[q, 2] - yi)^2 / 2
    end
    total = sum(dens)
    return total
end
@kernel matrix_row_port(X::AbstractMatrix{Float64}, y::AbstractVector{Float64},
                        rows::AbstractVector{Int}, key::AbstractVector{Int}) = begin
    dens = plate(rows, y) do q, yi
        -(sum(X[q, :]) - yi)^2 / 2
    end
    total = sum(dens)
    return total
end
@kernel vector_read_literal(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                            rows::AbstractVector{Int}, key::AbstractVector{Int}) = begin
    v = exp.(x)
    dens = plate(rows, y) do q, yi
        -(v[q, 1] - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

# Binding `w` splits `sum(w)` out of `anchor`, leaving `exp(v[1])` one closure.
@kernel bound_part_read(x::AbstractVector{Float64}, w::AbstractVector{Float64},
                        y::AbstractVector{Float64}, rows::AbstractVector{Int},
                        key::AbstractVector{Int}) = begin
    v = exp.(x)
    anchor = exp(v[1]) * sum(w) + v[1]
    dens = plate(rows, y) do q, yi
        -(anchor * v[q] - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

@testset "scalar index spelling and bounds" begin
    k = prepare(plate_partition(indexed, :dens; by = :key, locality = :error);
        have = (:x, :scale, :key, :partition), want = :partition_total)
    @test k([1.0, 2.0, 3.0], 0.5, [1, 2, 3], 2) == 3.5
    @test partition_footprint(k, [1.0, 2.0, 3.0], 0.5, [1, 2, 3], 2).x == [2, 3]
    @test_throws BoundsError k([1.0, 2.0], 0.5, [1, 2], 2)
end

@testset "scalar indexed demand through a predictor" begin
    beta = [0.2, -0.3]
    allocations, gradient_allocations, structures = Int[], Int[], Int[]
    for spec in (anchored, separate)
        empty!(allocations); empty!(gradient_allocations); empty!(structures)
        have = spec === anchored ? (:X, :beta, :y, :key, :partition) :
                                   (:X, :beta, :eta, :y, :key, :partition)
        partition = plate_partition(spec, :dens; by = :key, locality = :error)
        unbound = prepare(partition; have, want = :partition_total)
        for n in (8, 8000)
            m = spec === anchored ? n : n - 1
            X = [sin(i + j) / 3 for i in 1:m, j in 1:2]
            y, eta = cos.(1:n), sin.(1:n) ./ 10
            key = [1, 1, 2, 2, fill(3, n - 4)...]
            data = spec === anchored ? (; X, y, key) : (; X, eta, y, key)
            kernel = prepare(partition; have, want = :partition_total, bound = data)
            B = exp.(0.2 .* X)
            a = B * beta
            local_mean = spec === anchored ? a : exp.(eta)
            mu = exp(a[1]) .+ (a[2] - a[1]) .+ local_mean
            for label in (1, 2, 3)
                rows = findall(==(label), key)
                @test kernel(beta, label) ≈ -sum(abs2, mu[rows] - y[rows]) / 2
            end
            args = spec === anchored ? (X, beta, y, key, 2) : (X, beta, eta, y, key, 2)
            @test unbound(args...) ≈ kernel(beta, 2)
            footprint = partition_footprint(unbound, args...)
            rows = spec === anchored ? [1, 2, 3, 4] : [1, 2]
            @test footprint.X == [rows; m .+ rows]
            @test footprint.beta === :whole
            @test footprint.y == [3, 4]
            spec === separate && @test footprint.eta == [3, 4]
            report = partition_locality(kernel)
            @test all(e.status !== :whole for e in report)
            @test all(e.status in (:scalar, :rows) for e in report
                      if e.value in (:weight, :shift))
            @test all(e.status === :rows for e in report if e.value in (:a, :shared))
            kernel(beta, 2)
            push!(allocations, @allocated kernel(beta, 2))
            push!(structures, astsize(code_expr(kernel)))
            ad = prepare_ad(kernel, AutoEnzyme(; mode = Enzyme.Reverse), beta, 2; active = :beta)
            value, gradient = ad_value_and_gradient(ad, beta, 2)
            sensitivity = exp(a[1]) .* B[1, :] .+ B[2, :] .- B[1, :]
            expected = zeros(2)
            for row in (3, 4)
                derivative = spec === anchored ? sensitivity + B[row, :] : sensitivity
                expected .-= (mu[row] - y[row]) .* derivative
            end
            @test value ≈ kernel(beta, 2)
            @test gradient ≈ expected
            ad_value_and_gradient!(ad, gradient, beta, 2)
            push!(gradient_allocations, @allocated ad_value_and_gradient!(ad, gradient, beta, 2))
        end
        @test allocations[1] == allocations[2]
        @test gradient_allocations[2] <= gradient_allocations[1] + 1024
        @test structures[1] == structures[2]
    end
end

# The shared right factor is a transposed broadcast over an abstract vector.
# Its element type must survive to the scalar ratio, without granting matrix
# transposes a row form (their rows would read columns of the source).
@kernel scaled_right_factor(Z::AbstractMatrix{Float64}, scale::AbstractVector{Float64},
                            L::Matrix{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    right = transpose(scale .* L)
    B = Z * right
    c = exp.(B[:, 1])
    d = exp.(B[:, 2])
    ratio = c[1] / d[1]
    dens = plate(x, y) do xi, yi
        -(yi - sin(ratio * xi))^2 / 2
    end
    total = sum(dens)
    return total
end

function scaled_right_oracle(Z, scale, L, x, y, i)
    B = Z * transpose(scale .* L)
    ratio = exp(B[1, 1]) / exp(B[1, 2])
    -(y[i] - sin(ratio * x[i]))^2 / 2
end

@testset "one element of a matrix is read at its row" begin
    have = (:Z, :s, :y, :rows, :key, :partition)
    partition = plate_partition(matrix_read, :dens; by = :key, locality = :error)
    kernel = prepare(partition; have, want = :partition_total)
    status = Dict(e.value => e.status for e in partition_locality(kernel))
    @test status[:b] === :rows
    allocations, structures = Int[], Int[]
    for n in (6, 6000)
        Z = [sin(i + 2j) / 3 for i in 1:n, j in 1:2]
        rows = [4, n, 2, 4]
        y = [0.3, -0.2, 0.1, 0.4]
        key = [1, 1, 2, 2]
        s = 0.4
        element(i) = Z[i, 1] * s + Z[i, 2]
        anchor = exp(element(1))
        expected(k) = sum(-(anchor * element(rows[j]) - y[j])^2 / 2 for j in findall(==(k), key))
        @test kernel(Z, s, y, rows, key, 1) ≈ expected(1)
        @test kernel(Z, s, y, rows, key, 2) ≈ expected(2)
        # Row 1 for the anchor and the cells' own rows, whatever the data's size.
        linear(i) = LinearIndices(Z)[i, :]
        read = partition_footprint(kernel, Z, s, y, rows, key, 1).Z
        @test read == sort!(vcat(linear(1), linear(4), linear(n)))
        bound = prepare(partition; have, want = :partition_total, bound = (; y, rows, key))
        bound(Z, s, 1)
        push!(allocations, @allocated bound(Z, s, 1))
        push!(structures, astsize(code_expr(bound)))
        derivative = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse), Z, s, 1; active = :Z)
        value, gradient = ad_value_and_gradient(derivative, Z, s, 1)
        @test value ≈ expected(1)
        analytic = zeros(size(Z))
        for j in findall(==(1), key)
            residual = -(anchor * element(rows[j]) - y[j])
            analytic[1, :] .+= residual * anchor * element(rows[j]) .* [s, 1.0]
            analytic[rows[j], :] .+= residual * anchor .* [s, 1.0]
        end
        @test gradient ≈ analytic atol = 1e-12
    end
    @test structures[1] == structures[2]
    @test allocations[1] == allocations[2]
    # A row outside the matrix still throws.
    Z = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test_throws BoundsError kernel(Z, 0.1, [0.0], [4], [1], 1)
end

@testset "one element of a port, of bound data and of a vector" begin
    X = [sin(i + 2j) for i in 1:5, j in 1:2]
    y, rows, key = [0.2, -0.1, 0.4], [4, 2, 4], [1, 1, 2]
    expected(k) = sum(-(X[rows[j], 2] - y[j])^2 / 2 for j in findall(==(k), key))
    have = (:X, :y, :rows, :key, :partition)
    partition = plate_partition(matrix_read_port, :dens; by = :key, locality = :error)
    live = prepare(partition; have, want = :partition_total)
    @test live(X, y, rows, key, 1) ≈ expected(1)
    @test partition_footprint(live, X, y, rows, key, 1).X ==
          sort!(vcat(LinearIndices(X)[4, :], LinearIndices(X)[2, :]))
    bound = prepare(partition; have, want = :partition_total, bound = (; X, y, rows, key))
    @test bound(1) ≈ expected(1)
    @test bound(2) ≈ expected(2)
    # A selected row of a matrix port reports that row's positions, every column.
    rowwise = prepare(plate_partition(matrix_row_port, :dens; by = :key, locality = :error);
        have, want = :partition_total)
    @test rowwise(X, y, rows, key, 1) ≈
          sum(-(sum(X[rows[j], :]) - y[j])^2 / 2 for j in findall(==(1), key))
    @test partition_footprint(rowwise, X, y, rows, key, 1).X ==
          sort!(vcat(LinearIndices(X)[4, :], LinearIndices(X)[2, :]))

    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    vector = prepare(plate_partition(vector_read_literal, :dens; by = :key, locality = :error);
        have = (:x, :y, :rows, :key, :partition), want = :partition_total)
    @test vector(x, y, rows, key, 1) ≈
          sum(-(exp(x[rows[j]]) - y[j])^2 / 2 for j in findall(==(1), key))
    @test partition_footprint(vector, x, y, rows, key, 1).x == [2, 4]
end

@testset "a scalar read in a part split out by binding data" begin
    allocations, structures = Int[], Int[]
    for n in (6, 6000)
        x = [sin(i) / 3 for i in 1:n]
        w, y, rows, key = [0.5, 0.25], [0.1, -0.2, 0.3], [3, n, 3], [1, 1, 2]
        kernel = prepare(plate_partition(bound_part_read, :dens; by = :key, locality = :error);
            have = (:x, :w, :y, :rows, :key, :partition), want = :partition_total,
            bound = (; w, y, rows, key))
        v = exp.(x)
        anchor = exp(v[1]) * sum(w) + v[1]
        expected = sum(-(anchor * v[rows[j]] - y[j])^2 / 2 for j in 1:2)
        @test kernel(x, 1) ≈ expected
        @test partition_footprint(kernel, x, 1).x == [1, 3, n]
        status = Dict(e.value => e.status for e in partition_locality(kernel))
        @test status[:v] === :rows
        push!(allocations, @allocated kernel(x, 1))
        push!(structures, astsize(code_expr(kernel)))
        derivative = prepare_ad(kernel, AutoEnzyme(; mode = Enzyme.Reverse), x, 1; active = :x)
        value, gradient = ad_value_and_gradient(derivative, x, 1)
        @test value ≈ expected
        analytic = zeros(n)
        slope = exp(v[1]) * sum(w) + 1
        for j in 1:2
            residual = -(anchor * v[rows[j]] - y[j])
            analytic[1] += residual * v[rows[j]] * slope * v[1]
            analytic[rows[j]] += residual * anchor * v[rows[j]]
        end
        @test gradient ≈ analytic atol = 1e-10
    end
    @test structures[1] == structures[2]
    @test allocations[1] == allocations[2]
end

@testset "an element of a matrix without a row form stays refused" begin
    have = (:Z, :y, :rows, :key, :partition)
    @test_throws PartitionLocalityError prepare(
        plate_partition(matrix_read_reversed, :dens; by = :key, locality = :error);
        have, want = :partition_total)
    kernel = prepare(plate_partition(matrix_read_reversed, :dens; by = :key, locality = :ignore);
        have, want = :partition_total)
    Z = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test kernel(Z, [0.5], [1], [1], 1) ≈ -(reverse(Z; dims = 1)[1, 2] - 0.5)^2 / 2
end

@testset "scalar type through a shared transposed broadcast" begin
    have = (:Z, :scale, :L, :x, :y, :i)
    k = prepare(plate_cell(scaled_right_factor, :dens; index=:i, locality=:error);
                have, want=:dens_cell)
    structures = Int[]
    scale, L = [0.5, 0.8], [1.0 0.0; 0.2 0.9]
    for n in (6, 120)
        Z = [sin(i + j) / 4 for i in 1:n, j in 1:2]
        x, y = cos.(1:n), sin.(1:n)
        bound = prepare(plate_cell(scaled_right_factor, :dens; index=:i, locality=:error);
            have, want=:dens_cell, bound=(; Z, L, x, y))
        push!(structures, astsize(code_expr(bound)))
        for i in (1, n)
            expected = scaled_right_oracle(Z, scale, L, x, y, i)
            @test k(Z, scale, L, x, y, i) ≈ expected
            @test bound(scale, i) ≈ expected
            fp = partition_footprint(k, Z, scale, L, x, y, i)
            @test fp.Z == [1, n + 1]
            @test fp.x == [i] && fp.y == [i]
            ad = prepare_ad(bound, AutoEnzyme(; mode=Enzyme.Reverse), scale, i; active=:scale)
            value, gradient = ad_value_and_gradient(ad, scale, i)
            @test value ≈ expected
            for j in eachindex(scale)
                plus, minus = copy(scale), copy(scale)
                plus[j] += 1e-6
                minus[j] -= 1e-6
                fd = (scaled_right_oracle(Z, plus, L, x, y, i) -
                      scaled_right_oracle(Z, minus, L, x, y, i)) / 2e-6
                @test gradient[j] ≈ fd atol=2e-8 rtol=2e-6
            end
        end
    end
    @test structures[1] == structures[2]
    @test all(e.status !== :whole for e in partition_locality(k))
end

@kernel absent_scalar(u::AbstractVector{Float64}, y::AbstractVector{Float64},
                      extra::AbstractVector{Float64}, key::AbstractVector{Int},
                      extra_key::AbstractVector{Int}, anchor::Int) = begin
    m = log.(u)
    one = exp(m[anchor]) + m[anchor]
    a = plate(extra, one) do yi, mu
        -(yi - mu)^2 / 2
    end
    b = plate(y, m) do yi, mu
        -(yi - mu)^2 / 2
    end
end

@kernel absent_factor(u::AbstractVector{Float64}, Z::AbstractMatrix{Float64},
                      L::AbstractMatrix{Float64}, rows::AbstractVector{Int},
                      y::AbstractVector{Float64}, extra::AbstractVector{Float64},
                      key::AbstractVector{Int}, extra_key::AbstractVector{Int}) = begin
    raw = u[3:4]
    scale = log.(raw)
    F = scale .* L
    R = Z * F'
    mu = R[rows, 1] .+ R[rows, 2]
    a = plate(extra, mu) do yi, m
        -(yi - m)^2 / 2
    end
    b = plate(y, u[1:2]) do yi, m
        -(yi - m)^2 / 2
    end
end

_absent_factor_add(a, b) = a + b

@kernel absent_factor_capture(u::AbstractVector{Float64}, Z::AbstractMatrix{Float64},
                      L::AbstractMatrix{Float64}, rows::AbstractVector{Int},
                      y::AbstractVector{Float64}, extra::AbstractVector{Float64},
                      key::AbstractVector{Int}, extra_key::AbstractVector{Int}) = begin
    raw = u[3:4]
    scale = log.(raw)
    F = scale .* L
    R = Z * F'
    ma = R[rows, 1]
    mb = R[rows, 2]
    a = plate(extra, eachindex(extra)) do yi, i
        -(yi - _absent_factor_add(ma[i], mb[i]))^2 / 2
    end
    b = plate(y, u[1:2]) do yi, m
        -(yi - m)^2 / 2
    end
end

@testset "empty observation portions skip exclusive whole factors" begin
    structures = Dict{Any,Vector{Int}}()
    for model in (absent_factor, absent_factor_capture), n in (2, 200)
        spec = plate_partition(model, (:a, :b);
            by = (a = :extra_key, b = :key), locality = :error)
        bound = (; Z = [0.2 -0.1; 0.4 0.3], L = [1.0 0.0; 0.3 0.8],
            rows = repeat([1, 2], n ÷ 2), y = [0.1, 0.2],
            extra = fill(0.3, n), key = [1, 2], extra_key = fill(1, n))
        q = prepare(spec; want = :partition_total, bound)
        push!(get!(structures, model, Int[]), astsize(code_expr(q)))
        u = [0.2, -0.1, 1.2, 1.4]
        original = copy(u)
        for label in (1, 2)
            R = bound.Z * (log.(u[3:4]) .* bound.L)'
            mu = R[bound.rows, 1] .+ R[bound.rows, 2]
            expected = -(u[label] - bound.y[label])^2 / 2
            label == 1 && (expected -= sum(abs2, mu .- bound.extra) / 2)
            @test q(u, label) ≈ expected
            @test partition_footprint(q, u, label).u == (label == 1 ? [1, 3, 4] : [2])
            ad = prepare_ad(q, AutoEnzyme(; mode = Enzyme.Reverse), u, label; active = :u)
            value, gradient = ad_value_and_gradient(ad, u, label)
            @test value ≈ expected
            for j in eachindex(u)
                plus, minus = copy(u), copy(u)
                plus[j] += 1e-6
                minus[j] -= 1e-6
                @test gradient[j] ≈ (q(plus, label) - q(minus, label)) / 2e-6 atol=2e-8 rtol=2e-6
            end
        end
        @test u == original
        u[3] = -1.0
        @test q(u, 2) ≈ -(u[2] - bound.y[2])^2 / 2
        ad = prepare_ad(q, AutoEnzyme(; mode = Enzyme.Reverse), u, 2; active = :u)
        value, gradient = ad_value_and_gradient(ad, u, 2)
        @test value ≈ q(u, 2)
        @test gradient ≈ [0.0, bound.y[2] - u[2], 0.0, 0.0]
        @test_throws DomainError q(u, 1)
        whole = prepare(spec; want = (:partition_total, :F), bound)
        @test_throws DomainError whole(u, 2)
    end
    @test all(s -> s[1] == s[2], values(structures))
end

# Two partition sums sharing a pushed value (`s`) lower with one cell emitter.
# Sum `a` has cells only in label 1 and alone reads a whole factor (`F`) and a
# scalar (`tau`) of label-1 coordinates (snag `empty-vector regression`).
@kernel grouped_absent_factor(u::AbstractVector{Float64}, Z::AbstractMatrix{Float64},
                              L::AbstractMatrix{Float64}, rows::AbstractVector{Int},
                              arow::AbstractVector{Int}, y::AbstractVector{Float64},
                              extra::AbstractVector{Float64}, key::AbstractVector{Int},
                              extra_key::AbstractVector{Int}) = begin
    s = exp.(u[1:2])
    raw = u[3:4]
    scale = log.(raw)
    F = scale .* L
    R = Z * F'
    mu = R[rows, 1] .+ R[rows, 2]
    tau::Float64 = exp(u[5])
    sa = s[arow]
    a = plate(extra, mu, sa, tau) do yi, m, si, t
        density::Float64 = -(yi - m - si)^2 / 2 * t
        density
    end
    b = plate(y, s) do yi, si
        density::Float64 = -(yi - si)^2 / 2
        density
    end
end

# The rows `a` reads of the shared `X` depend on `a`'s own whole factor `w`.
@kernel grouped_absent_rows(u::AbstractVector{Float64}, Z::AbstractMatrix{Float64},
                            L::AbstractMatrix{Float64}, rows::AbstractVector{Int},
                            arow::AbstractVector{Int}, y::AbstractVector{Float64},
                            extra::AbstractVector{Float64}, key::AbstractVector{Int},
                            extra_key::AbstractVector{Int}) = begin
    s = exp.(u[1:2])
    X = Z .* s
    w::Vector{Float64} = log.(u[3:4]) .+ u[5]
    m = exp.(X * w)
    ma = m[rows]
    a = plate(extra, ma) do yi, mi
        density::Float64 = -(yi - mi)^2 / 2
        density
    end
    b = plate(y, X[:, 1]) do yi, x
        density::Float64 = -(yi - x)^2 / 2
        density
    end
end

function _grouped_absent_expected(model, bound, u, label)
    s = exp.(u[1:2])
    if model === grouped_absent_factor
        value = -(s[label] - bound.y[label])^2 / 2
        label == 1 || return value
        R = bound.Z * (log.(u[3:4]) .* bound.L)'
        mu = R[bound.rows, 1] .+ R[bound.rows, 2]
        return value - sum(abs2, bound.extra .- mu .- s[bound.arow]) / 2 * exp(u[5])
    end
    X = bound.Z .* s
    value = -(X[label, 1] - bound.y[label])^2 / 2
    label == 1 || return value
    m = exp.(X * (log.(u[3:4]) .+ u[5]))
    value - sum(abs2, bound.extra .- m[bound.rows]) / 2
end

@testset "grouped partition sums skip an empty member's exclusive work" begin
    structures = Dict{Any,Vector{Int}}()
    for model in (grouped_absent_factor, grouped_absent_rows), plan in (nothing, :plan),
            n in (2, 200)
        # `a` reads both rows of `X`, so `u[2]` is shared in the second model.
        factor = model === grouped_absent_factor
        labels = Union{Missing,Int}[1, factor ? 2 : missing, 1, 1, 1]
        spec = plate_partition(model, (:a, :b); by = (a = :extra_key, b = :key),
            locality = :error, partitioned = (:u => labels,), plan)
        bound = (; Z = [0.2 -0.1; 0.4 0.3], L = [1.0 0.0; 0.3 0.8],
            rows = repeat([1, 2], n ÷ 2), arow = fill(1, n), y = [0.1, 0.2],
            extra = fill(0.3, n), key = [1, 2], extra_key = fill(1, n))
        u = [0.2, -0.1, 1.2, 1.4, 0.3]
        q = plan === nothing ? prepare(spec; want = :partition_total, bound) :
            prepare(spec; want = :partition_total, bound, exemplars = (u, 1, nothing))
        push!(get!(structures, (model, plan), Int[]), astsize(code_expr(q)))
        extra(label, u) = plan === nothing ? () : (partition_plan(q, u, label, nothing),)
        footprint(label, u) = plan === nothing ? partition_footprint(q, u, label).u :
            partition_footprint(q, u, label, nothing).u
        original = copy(u)
        for label in (1, 2)
            expected = _grouped_absent_expected(model, bound, u, label)
            @test q(u, label, extra(label, u)...) ≈ expected
            # Label 2 reads only its own coordinate of `u`, so the strict
            # separability check accepts it.
            @test footprint(label, u) ==
                  (label == 2 ? [2] : factor ? [1, 3, 4, 5] : [1, 2, 3, 4, 5])
            plan === nothing && @test partition_separability(q, u, label).u == footprint(label, u)
            ad = prepare_ad(q, AutoEnzyme(; mode = Enzyme.Reverse), u, label,
                            extra(label, u)...; active = :u)
            value, gradient = ad_value_and_gradient(ad, u, label, extra(label, u)...)
            @test value ≈ expected
            for j in eachindex(u)
                plus, minus = copy(u), copy(u)
                plus[j] += 1e-6
                minus[j] -= 1e-6
                @test gradient[j] ≈ (q(plus, label, extra(label, plus)...) -
                                     q(minus, label, extra(label, minus)...)) / 2e-6 atol=2e-8 rtol=2e-6
            end
        end
        @test u == original
        # An empty member evaluates none of its own work, even invalid work.
        u[3] = -1.0
        @test q(u, 2, extra(2, u)...) ≈ _grouped_absent_expected(model, bound, u, 2)
        @test_throws DomainError q(u, 1, extra(1, u)...)
    end
    @test all(s -> s[1] == s[2], values(structures))
end

# Sums `a` and `b` share a whole factor of label-1 coordinates and both have
# cells only in label 1; `c` has cells in both labels and shares the pushed `s`
# with them, so all three lower with one emitter.
@kernel grouped_shared_factor(u::AbstractVector{Float64}, Z::AbstractMatrix{Float64},
                              L::AbstractMatrix{Float64}, rows::AbstractVector{Int},
                              arow::AbstractVector{Int}, ya::AbstractVector{Float64},
                              yb::AbstractVector{Float64}, yc::AbstractVector{Float64},
                              akey::AbstractVector{Int}, bkey::AbstractVector{Int},
                              ckey::AbstractVector{Int}) = begin
    s = exp.(u[1:2])
    raw = u[3:4]
    scale = log.(raw)
    F = scale .* L
    R = Z * F'
    ma = R[rows, 1] .+ s[arow]
    mb = R[rows, 2] .- s[arow]
    a = plate(ya, ma) do y, m
        density::Float64 = -(y - m)^2 / 2
        density
    end
    b = plate(yb, mb) do y, m
        density::Float64 = -(y - m)^2 / 2
        density
    end
    c = plate(yc, s) do y, si
        density::Float64 = -(y - si)^2 / 2
        density
    end
end

@testset "work several grouped sums share follows its readers" begin
    factor_bound = (; Z = [0.2 -0.1; 0.4 0.3], L = [1.0 0.0; 0.3 0.8],
        rows = repeat([1, 2], 2), arow = fill(1, 4), ya = fill(0.3, 4), yb = fill(-0.2, 4),
        yc = [0.1, 0.2], akey = fill(1, 4), bkey = fill(1, 4), ckey = [1, 2])
    function factor_expected(u, label)
        d = factor_bound
        s = exp.(u[1:2])
        value = -(d.yc[label] - s[label])^2 / 2
        label == 1 || return value
        R = d.Z * (log.(u[3:4]) .* d.L)'
        value - sum(abs2, d.ya .- (R[d.rows, 1] .+ s[1])) / 2 -
            sum(abs2, d.yb .- (R[d.rows, 2] .- s[1])) / 2
    end
    cases = ((grouped_shared_factor, (:a, :b, :c), (a = :akey, b = :bkey, c = :ckey),
              factor_bound, factor_expected),)
    for (model, plates, by, bound, expected) in cases, plan in (nothing, :plan)
        spec = plate_partition(model, plates; by, locality = :error,
            partitioned = (:u => Union{Missing,Int}[1, 2, 1, 1],), plan)
        u = [0.2, -0.1, 1.2, 1.4]
        q = plan === nothing ? prepare(spec; want = :partition_total, bound) :
            prepare(spec; want = :partition_total, bound, exemplars = (u, 1, nothing))
        extra(label, u) = plan === nothing ? () : (partition_plan(q, u, label, nothing),)
        footprint(label, u) = plan === nothing ? partition_footprint(q, u, label).u :
            partition_footprint(q, u, label, nothing).u
        for label in (1, 2)
            @test q(u, label, extra(label, u)...) ≈ expected(u, label)
            # Label 2 has no cell of the sums that read coordinates 3 and 4.
            @test footprint(label, u) == (label == 1 ? [1, 3, 4] : [2])
            plan === nothing && @test partition_separability(q, u, label).u == footprint(label, u)
            ad = prepare_ad(q, AutoEnzyme(; mode = Enzyme.Reverse), u, label,
                            extra(label, u)...; active = :u)
            value, gradient = ad_value_and_gradient(ad, u, label, extra(label, u)...)
            @test value ≈ expected(u, label)
            for j in eachindex(u)
                plus, minus = copy(u), copy(u)
                plus[j] += 1e-6
                minus[j] -= 1e-6
                @test gradient[j] ≈ (q(plus, label, extra(label, plus)...) -
                                     q(minus, label, extra(label, minus)...)) / 2e-6 atol=2e-8 rtol=2e-6
            end
        end
    end
    # The shared factor is not evaluated for label 2, even when invalid there.
    spec = plate_partition(grouped_shared_factor, (:a, :b, :c);
        by = (a = :akey, b = :bkey, c = :ckey), locality = :error)
    q = prepare(spec; want = :partition_total, bound = factor_bound)
    v = [0.2, -0.1, -1.0, 1.4]
    @test q(v, 2) ≈ factor_expected(v, 2)
    @test_throws DomainError q(v, 1)
end

@testset "empty observation portions do not demand scalar dependencies" begin
    spec = plate_partition(absent_scalar, (:a, :b);
        by = (a = :extra_key, b = :key), locality = :error)
    structures = Int[]
    for n in (3, 300)
        bound = (; y = fill(0.1, n), extra = [0.4], key = collect(1:n),
            extra_key = [1], anchor = 1)
        q = prepare(spec; want = :partition_total, bound)
        push!(structures, astsize(code_expr(q)))
        u = fill(1.2, n)
        for label in (1, 2, n)
            expected = -(log(u[label]) - 0.1)^2 / 2
            label == 1 && (expected -= (u[1] + log(u[1]) - 0.4)^2 / 2)
            @test q(u, label) ≈ expected
            @test partition_footprint(q, u, label).u == [label]
        end
        # An absent factor must skip arithmetic that would fail if evaluated.
        u[1] = -1.0
        @test q(u, 2) ≈ -(log(u[2]) - 0.1)^2 / 2
        @test_throws DomainError q(u, 1)
        # Selecting one total retains the label universe of both observations.
        # The remaining partition can therefore still have an empty portion.
        extra_total = only(v for v in values(spec.graph.values) if v.name === :_partition_a)
        partial = prepare(spec.graph;
            have = Tuple(spec[name] for name in (:u, :y, :extra, :key, :extra_key, :anchor, :partition)),
            want = extra_total, bound = [spec[name] => getproperty(bound, name) for name in keys(bound)])
        @test partial(u, 2) == 0.0
        @test isempty(partition_footprint(partial, u, 2).u)
        @test_throws DomainError partial(u, 1)
        # The same applies to an invalid scalar index, not just arithmetic.
        invalid = prepare(spec; want = :partition_total,
            bound = merge(bound, (; anchor = n + 1)))
        @test invalid(u, 2) ≈ -(log(u[2]) - 0.1)^2 / 2
        @test_throws BoundsError invalid(abs.(u), 1)
        # A scalar requested as an output is still evaluated as a scalar.
        both = prepare(spec; want = (:partition_total, :one), bound)
        @test_throws DomainError both(u, 2)
        @test both(abs.(u), 2)[2] isa Float64
    end
    @test structures[1] == structures[2]
end
end
