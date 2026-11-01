module CellGatherColumnTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test, LinearAlgebra
using DifferentiationInterface: AutoEnzyme

const MODEL = quote
    a[1:4] .~ Normal.(0, 1)
    Z[1:n, 1:4] .~ Normal.(0, 1)
    B = Z * transpose(exp.(a) .* L)
    mu = .+(B[ids, 1] .* x1, B[ids, 2] .* x2,
            B[ids, 3] .* x3, B[ids, 4] .* x4)
    y .~ Normal.(mu, 1.0)
end

@testset "strict PPL cells keep multi-column gathered predictors local" begin
    data = (; n = 3, L = [1.0 0.0 0.0 0.0; 0.2 1.0 0.0 0.0;
        -0.1 0.3 1.0 0.0; 0.4 0.0 -0.2 1.0], ids = [2, 1, 2, 3],
        x1 = [0.2, 0.3, -0.4, 0.0], x2 = [0.0, -0.2, 0.1, 0.5],
        x3 = [0.4, -0.5, 0.0, 0.2], x4 = [0.1, 0.0, 0.6, -0.3],
        y = [0.1, 0.7, -0.2, 0.5])
    original = deepcopy(data)
    plan = lower_rkppl(MODEL, data; conditioned = (:y,), mod = @__MODULE__)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    u = sin.(1:built.layout.total) ./ 4
    function reference(v, i)
        values = Dict(zip(names, v))
        a = [values[Symbol("a.$j")] for j in 1:4]
        z = [values[Symbol("Z.$(data.ids[i]).$j")] for j in 1:4]
        x = [data.x1[i], data.x2[i], data.x3[i], data.x4[i]]
        mu = dot((exp.(a) .* data.L) * z, x)
        -log(2pi)/2 - (data.y[i] - mu)^2/2
    end
    q = prepare_cell_query(built, bound, :y; locality = :error)
    ad = prepare_ad(q.kernel, AutoEnzyme(; mode = Enzyme.Reverse), u, 1;
        active = :unconstrained)
    for v in (u, -u), i in (1, 4)
        expected = reference(v, i)
        @test q(v, i) ≈ expected
        value, gradient = ad_value_and_gradient(ad, v, i)
        @test value ≈ expected
        numeric = map(eachindex(v)) do j
            plus, minus = copy(v), copy(v)
            plus[j] += 1e-6
            minus[j] -= 1e-6
            (reference(plus, i) - reference(minus, i)) / 2e-6
        end
        @test gradient ≈ numeric atol = 1e-8 rtol = 1e-6
        support = partition_footprint(q.kernel, v, i).unconstrained
        expected_names = Set([Symbol("a.$j") for j in 1:4])
        union!(expected_names, (Symbol("Z.$(data.ids[i]).$j") for j in 1:4))
        @test Set(names[support]) == expected_names
    end
    @test all(e.status !== :whole for e in partition_locality(q.kernel))
    @test data == original
end
end
