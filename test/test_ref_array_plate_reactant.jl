using ReactiveKernels, Reactant, DifferentiationInterface, Test
import Enzyme
isdefined(@__MODULE__, :AuthoredPlateChains) ||
    include("fixtures/authored_plate_chains.jl")

module RefArrayPlateFixtures
using ReactiveKernels

@kernel broadcast_axes(q::Vector{Float64}, x, y) = begin
    pointwise = plate(x, Ref(q), y) do xi, whole, yi
        xi + sum(whole) * yi
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel mixed(q::Vector{Float64}) = begin
    pointwise = plate(q, Ref(q)) do qi, whole
        qi + sum(whole)
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel matrix_atom(q::Matrix{Float64}, x::Vector{Float64}) = begin
    pointwise = plate(Ref(q), x) do whole, xi
        sum(whole) * xi
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel scalar_atom(s::Float64, x::Vector{Float64}, y::Vector{Float64}) = begin
    middle = plate(x, Ref(s)) do xi, shared
        2 * xi + shared
    end
    pointwise = plate(y, middle, Ref(s)) do yi, mi, shared
        yi + mi^2 + shared * mi
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel columns(q::Vector{Float64}, x::Matrix{Float64}, weights::Vector{Float64}) = begin
    pointwise = plate(Ref(q), weights, eachcol(x)) do whole, weight, column
        weight * sum(column .* whole)
    end
    total::Float64 = sum(pointwise)
    return total
end
end

@testset "Ref array plates retain atomic parameters in Reactant" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q = [0.7, -0.3]
    rq = Reactant.to_rarray(q)
    @testset "N=$n" for n in (8, 32, 1, 0)
        x = n == 0 ? Float64[] : n == 1 ? [0.0] :
            collect(range(-1.0, 1.0; length = n))
        y = fill(0.3, n)
        middle = 2 .* x .+ q[1]
        pointwise = y .+ middle.^2 .+ q[2] .* middle
        total = sum(pointwise)
        # This is the reporter's exact graph: the only live HAVE is Ref(q),
        # whose length is deliberately different from the bound lane count.
        k = prepare(AuthoredPlateChains.ref_atomic_chain; bound = (; x, y))
        compiled = Reactant.@compile sync = true k(rq)
        @test Float64(compiled(rq)) ≈ total
        @test k(q) ≈ total

        both = prepare(AuthoredPlateChains.ref_atomic_chain;
            want = (:pointwise, :total), bound = (; x, y))
        compiled_both = Reactant.@compile sync = true both(rq)
        values, value = compiled_both(rq)
        @test Array(values) ≈ pointwise
        @test Float64(value) ≈ total
        demanded = prepare(AuthoredPlateChains.ref_atomic_chain;
            want = :middle, bound = (; x, y))
        compiled_middle = Reactant.@compile sync = true demanded(rq)
        @test Array(compiled_middle(rq)) ≈ middle

        ad = prepare_ad(k, backend, q; active = :q)
        compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rq)
        value, gradient = compiled_ad(ad, rq)
        @test Float64(value) ≈ total
        @test Array(gradient) ≈ [sum(2 .* middle .+ q[2]), sum(middle)]

        # A compiled closure must keep q dynamic, not freeze its first payload.
        q2 = [-0.2, 0.5]
        rq2 = Reactant.to_rarray(q2)
        middle2 = 2 .* x .+ q2[1]
        value2, gradient2 = compiled_ad(ad, rq2)
        @test Float64(value2) ≈ sum(y .+ middle2.^2 .+ q2[2] .* middle2)
        @test Array(gradient2) ≈ [sum(2 .* middle2 .+ q2[2]), sum(middle2)]

        unbound = prepare(AuthoredPlateChains.ref_atomic_chain)
        rx, ry = Reactant.to_rarray.((x, y))
        compiled_unbound = Reactant.@compile sync = true unbound(rq, rx, ry)
        @test Float64(compiled_unbound(rq, rx, ry)) ≈ total
        scalar = prepare(RefArrayPlateFixtures.scalar_atom; bound = (; x, y))
        rs = Reactant.to_rarray(0.5)
        compiled_scalar = Reactant.@compile sync = true scalar(rs)
        scalar_middle = 2 .* x .+ 0.5
        @test Float64(compiled_scalar(rs)) ≈
            sum(y .+ scalar_middle.^2 .+ 0.5 .* scalar_middle)
    end

    @testset "broadcast axes and shared rank are independent" begin
        for (x, y) in (([1.0], collect(1.0:32)),
                      ((1.0,), collect(1.0:32)),
                      (reshape([1.0, 2.0], 2, 1), reshape(collect(1.0:3), 1, 3)))
            k = prepare(RefArrayPlateFixtures.broadcast_axes;
                want = :pointwise, bound = (; x, y))
            compiled = Reactant.@compile sync = true k(rq)
            actual = Array(compiled(rq))
            reference = x .+ sum(q) .* y
            @test size(actual) == size(reference)
            @test actual ≈ reference
        end
        qmatrix = reshape(collect(0.1:0.1:0.6), 2, 3)
        rmatrix = Reactant.to_rarray(qmatrix)
        x = collect(1.0:32)
        k = prepare(RefArrayPlateFixtures.matrix_atom; bound = (; x))
        compiled = Reactant.@compile sync = true k(rmatrix)
        @test Float64(compiled(rmatrix)) ≈ sum(qmatrix) * sum(x)
        ad = prepare_ad(k, backend, qmatrix; active = :q)
        compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rmatrix)
        _, gradient = compiled_ad(ad, rmatrix)
        @test Array(gradient) ≈ fill(sum(x), size(qmatrix))
    end

    @testset "one array is both elementwise and atomic" begin
        for n in (8, 32)
            qmixed = collect(1.0:n) ./ n
            rmixed = Reactant.to_rarray(qmixed)
            k = prepare(RefArrayPlateFixtures.mixed)
            compiled = Reactant.@compile sync = true k(rmixed)
            @test Float64(compiled(rmixed)) ≈ (n + 1) * sum(qmixed)
            ad = prepare_ad(k, backend, qmixed; active = :q)
            compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rmixed)
            _, gradient = compiled_ad(ad, rmixed)
            @test Array(gradient) ≈ fill(n + 1, n)
        end
    end

    @testset "eachcol keeps its structural batch with a leading Ref" begin
        for n in (8, 32)
            x = reshape(collect(1.0:(2n)) ./ n, 2, n)
            weights = collect(1.0:n) ./ n
            rx = Reactant.to_rarray(x)
            k = prepare(RefArrayPlateFixtures.columns; bound = (; weights))
            compiled = Reactant.@compile sync = true k(rq, rx)
            reference_gradient = vec(sum(x .* reshape(weights, 1, n); dims = 2))
            @test Float64(compiled(rq, rx)) ≈ sum(q .* reference_gradient)
            @test k(q, x) ≈ sum(q .* reference_gradient)
            ad = prepare_ad(k, backend, q, x; active = :q)
            compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rq, rx)
            _, gradient = compiled_ad(ad, rq, rx)
            @test Array(gradient) ≈ reference_gradient
        end
    end
end
