using ReactiveKernels, DifferentiationInterface, Enzyme, Test
isdefined(@__MODULE__, :NestedPlates) || include("fixtures/nested_plates.jl")

@testset "Transparent native nested plate regions" begin
    N = NestedPlates
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    groups = [[0.2, 0.7], Float64[], [-1.2]]
    original = deepcopy(groups)
    k = prepare(N.visible_observations)
    outer = only(filter(r -> r.op isa ReactiveKernels._AuthoredPlateOp, k.plan.recipes))
    inner = only(filter(r -> r.op isa ReactiveKernels._AuthoredPlateOp,
                        plate_body(outer).recipes))
    @test length(plate_body(inner).recipes) > 0
    @test occursin("log(scale)", string(only(plate_body(inner).recipes).source))
    @test N.head_count(code_expr(k), :for) == 2
    @test !occursin("similar", string(code_expr(k)))

    # The authored source is complete, and replays in a fresh module without
    # the producer's prepared kernels, operation tables or private bindings.
    replay = Module(gensym(:NestedPlateReplay))
    Core.eval(replay, :(using ReactiveKernels))
    printed = sprint(Base.show_unquoted, Meta.parse(N.NORMAL_SOURCE))
    Core.eval(replay, Meta.parse(printed))
    replay_k = prepare(getfield(replay, :visible_observations))
    generated = sprint(Base.show_unquoted,
        ReactiveKernels._readable_expr(code_expr(k), k); context=:limit => false)
    @test !occursin("__ops__[", generated)
    @test occursin("log(scale)", generated)

    for data in (groups, [Float64[]], Vector{Float64}[], [[1.1], [0.3, -0.4, 0.8]],
                 [(0.25, 0.9), (), (0.7,)], [(), (0.25, 0.9), (0.7,)],
                 [(0.25, 0.9), (0.7,), ()], Tuple{Vararg{Float64}}[(), ()],
                 Tuple{Vararg{Float64}}[])
        n = sum(length, data; init=0)
        sq = sum(v -> sum(abs2, v; init=0.0), data; init=0.0)
        for sigma in (0.6, 1.3)
            expected = -0.5*n*log(2*pi) - n*log(sigma) - 0.5*sq/sigma^2
            gradient = -n/sigma + sq/sigma^3
        @test k(data, sigma) ≈ expected
            @test replay_k(data, sigma) ≈ expected
            @test eltype(prepare(N.visible_observations; want=:group_logdensity)(data, sigma)) == Float64
            bound = prepare(N.visible_observations; bound=(; observation_groups=data))
            @test bound(sigma) ≈ expected
            for (reader, args) in ((k, (data, sigma)), (bound, (sigma,)))
                ad = prepare_ad(reader, backend, args...; active=:scale)
                value, grad = ad_value_and_gradient(ad, args...)
                @test value ≈ expected
                @test grad ≈ gradient
            end
        end
    end
    @test groups == original
    @test k([(0.2, 0.7), (-1.2, 0.1)], 1.3) ≈
          k([[0.2, 0.7], [-1.2, 0.1]], 1.3)
    @test k([(0.25, 0.9), (), (0.7,)], 1.3) ≈ -3.9470149018922305
    @test k([(1, 2), (), (3,)], 1.3) ≈ k([[1, 2], Int[], [3]], 1.3)

    for spec in (N.cross_product, N.inline_product, N.composed_product)
        product = prepare(spec)
        @test product([0.3, -0.7], [2.0, 4.0]) ≈ [1.8, -4.2]
        @test product([0.3, -0.7], Float64[]) == zeros(2)
        @test product(Float64[], [2.0, 4.0]) == Float64[]
        @test N.head_count(code_expr(product), :for) == 2
    end

    deep = prepare(N.three_levels)
    @test deep([[[1.0], Float64[]], [[2.0, 3.0]]], 2.0) == 12.0
    @test deep(Vector{Vector{Float64}}[], 2.0) == 0.0
    @test deep([[(), (1.0,)], [(2.0, 3.0), ()]], 2.0) == 12.0
    @test deep([Tuple{Vararg{Float64}}[(), ()]], 2.0) == 0.0
    @test N.head_count(code_expr(deep), :for) == 3
    @test ad_gradient(prepare_ad(deep, backend, [[[1.0]], [[2.0, 3.0]]], 2.0;
                                active=:scale), [[[1.0]], [[2.0, 3.0]]], 2.0) == 6.0

    rectangular = prepare(N.rectangular)
    for (rows, cols) in ((0, 3), (3, 0), (2, 3), (17, 11))
        X = reshape(sin.(1:rows*cols), rows, cols)
        @test rectangular(X, 0.7) ≈ 0.7*sum(X)
        @test N.head_count(code_expr(rectangular), :for) == 2
        @test ad_gradient(prepare_ad(rectangular, backend, X, 0.7; active=:scale),
                          X, 0.7) ≈ sum(X)
    end
    @test prepare(N.nested_axes)([reshape([1.0, 2.0], 2, 1)], [1.0 3.0]) == [12.0]
    @test_throws DimensionMismatch prepare(N.nested_axes)([ones(2, 1)], ones(3, 1))

    guarded = prepare(N.guarded)
    # The inactive log arm is outside its scalar domain for x = -3.
    @test guarded([[-3.0, 0.3], Float64[]], 1.2) ≈ 3*1.2 + log(1.5)
    @test ad_gradient(prepare_ad(guarded, backend, [[-3.0, 0.3]], 1.2; active=:scale),
                      [[-3.0, 0.3]], 1.2) ≈ 3 + 1/1.5
    @test guarded([(), (-3.0,)], 1.2) ≈ 3*1.2
    @test guarded(Tuple{Vararg{Float64}}[(), ()], -1.2) === 0.0

    # Size changes re-use one graph/AST; neither loop is replicated, and a
    # total-only read allocates no pointwise array at either nesting level.
    for n in (3, 31)
        data = [collect(range(-1.0, 1.0; length=n)) for _ in 1:n]
        N.allocated(k, data, 1.3)
        @test N.allocated(k, data, 1.3) == 0
    end
end

@testset "Nested computed axes retain concrete result types in native Reverse" begin
    N = NestedPlates
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    k = prepare(N.indexed_loss)
    ast = deepcopy(code_expr(k))
    @test N.head_count(ast, :for) == 3 # nested projection, then the broadcast consumer
    for (n, m) in ((0, 3), (4, 0), (4, 3), (17, 9)), shift in (0.0, 0.3)
        matrix = reshape(sin.(1.0:n*m), n, m)
        rates = collect(range(0.2, 1.0; length=m))
        groups = [isodd(i) ? 1 : 2 for i in 1:n]
        targets = cos.(1.0:n)
        parameters = [0.8 + shift; 0.4 - shift; 0.15 + shift; sin.(1.0:2*m)]
        saved = deepcopy((matrix, rates, groups, targets, parameters))
        coefficients = reshape(parameters[4:end], 2, m)
        predictions = [parameters[3] + sum(matrix[i, j] * parameters[1] * exp(-parameters[2]*rates[j]) *
                           coefficients[groups[i], j] for j in 1:m; init=0.0) for i in 1:n]
        expected = sum(abs2, predictions - targets)
        gradient = zeros(length(parameters))
        gradient[3] = 2sum(predictions - targets)
        for i in 1:n, j in 1:m
            residual = predictions[i] - targets[i]
            feature = matrix[i, j] * exp(-parameters[2]*rates[j])
            coefficient = coefficients[groups[i], j]
            gradient[1] += 2residual * feature * coefficient
            gradient[2] -= 2residual * feature * coefficient * parameters[1] * rates[j]
            gradient[3 + groups[i] + 2(j-1)] += 2residual * feature * parameters[1]
        end
        bound = prepare(N.indexed_loss; bound=(; matrix, rates, groups, targets))
        for (reader, args) in ((k, (matrix, rates, groups, targets, parameters)),
                               (bound, (parameters,)))
            @test reader(args...) ≈ expected
            ad = prepare_ad(reader, backend, args...; active=:parameters)
            for _ in 1:2
                value, actual = ad_value_and_gradient(ad, args...)
                @test value ≈ expected
                @test actual ≈ gradient
                @test (matrix, rates, groups, targets, parameters) == saved
            end
        end
        @test code_expr(k) == ast
    end
end

@testset "Nested plate endpoint caller scope" begin
    N = NestedPlates
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    replay = Module(gensym(:NestedEndpointReplay))
    Core.eval(replay, :(using ReactiveKernels))
    Core.eval(replay, Meta.parseall(N.ENDPOINT_SOURCE))
    normal(y, mu, sigma) = -0.5*log(2*pi) - log(sigma) - 0.5*((y-mu)/sigma)^2

    for name in (:object_observations, :method_observations, :computed_observations)
        spec = getfield(N, name)
        k = prepare(spec)
        replay_k = prepare(getfield(replay, name))
        outer = only(filter(r -> r.op isa ReactiveKernels._AuthoredPlateOp,
                            k.plan.recipes))
        inner = only(filter(r -> r.op isa ReactiveKernels._AuthoredPlateOp,
                            plate_body(outer).recipes))
        @test all(r -> !(r.op isa ReactiveKernels._AuthoredPlateOp),
                  plate_body(inner).recipes)
        @test any(r -> occursin("log(", string(r.source)), plate_body(inner).recipes)
        @test N.head_count(code_expr(k), :for) == 2
        @test !occursin("similar", string(code_expr(k)))
        ast = deepcopy(code_expr(k))
        for groups in ([[0.25, 0.9], Float64[], [0.7]],
                       [Float64[]], Vector{Float64}[],
                       [(0.25, 0.9), (), (0.7,)],
                       Tuple{Vararg{Float64}}[(), ()], Tuple{Vararg{Float64}}[],
                       [collect(range(-0.7, 0.9; length=17)), [0.2]])
            original = deepcopy(groups)
            for (mu, sigma) in ((0.1, 1.3), (-0.2, 0.7))
                n = sum(length, groups; init=0)
                residual_sum = sum(g -> sum(y -> y-mu, g; init=0.0), groups; init=0.0)
                sq = sum(g -> sum(y -> (y-mu)^2, g; init=0.0), groups; init=0.0)
                expected = sum(g -> sum(y -> normal(y, mu, sigma), g; init=0.0),
                               groups; init=0.0)
                @test k(groups, mu, sigma) ≈ expected
                @test replay_k(groups, mu, sigma) ≈ expected
                bound = prepare(spec; bound=(; groups))
                @test bound(mu, sigma) ≈ expected
                @test prepare(spec; want=:grouped)(groups, mu, sigma) ≈
                      [sum(y -> normal(y, mu, sigma), g; init=0.0) for g in groups]
                for (reader, args) in ((k, (groups, mu, sigma)), (bound, (mu, sigma)))
                    ad = prepare_ad(reader, backend, args...; active=(:mu, :sigma))
                    value, gradient = ad_value_and_gradient(ad, args...)
                    @test value ≈ expected
                    @test gradient[1] ≈ residual_sum/sigma^2
                    @test gradient[2] ≈ -n/sigma + sq/sigma^3
                end
                @test groups == original
                @test code_expr(k) == ast
            end
        end
        @test k([(0.25, 0.9), (0.7,)], 0.1, 1.3) ≈
              k([[0.25, 0.9], [0.7]], 0.1, 1.3)
        data = [0.25, 0.9, 0.7]
        @test k([view(data, 1:2), view(data, 3:3)], 0.1, 1.3) ≈
              sum(y -> normal(y, 0.1, 1.3), data)
    end

    guarded = prepare(N.guarded_observations)
    @test guarded([[0.2, 0.7], Float64[]], 0.1, 1.3) ≈
          normal(0.2, 0.1, 1.3) + normal(0.7, 0.1, 1.3)
    @test ad_gradient(prepare_ad(guarded, backend, [[0.2, 0.7]], 0.1, 1.3;
                                active=:sigma), [[0.2, 0.7]], 0.1, 1.3) ≈
          -2/1.3 + (0.1^2 + 0.6^2)/1.3^3
    # Selected negative-scale arm preserves Julia's real log domain error.
    @test_throws DomainError guarded([[-0.2]], 0.1, 1.3)
    deep = prepare(N.deep_object_observations)
    @test N.head_count(code_expr(deep), :for) == 3
    @test deep([[[0.2], Float64[]], [[0.7]]], 0.1, 1.3) ≈
          normal(0.2, 0.1, 1.3) + normal(0.7, 0.1, 1.3)
    @test deep(Vector{Vector{Float64}}[], 0.1, 1.3) == 0.0
    scanned = prepare(N.scanned_object_observations)
    @test scanned([[0.2, 0.7], Float64[]], 0.1, 1.3) ≈
          normal(0.2, 0.1, 1.3) + normal(0.7, 0.1, 1.3)
    @test scanned(Vector{Float64}[], 0.1, 1.3) == 0.0
end

@testset "Declared empty tuple plate results" begin
    N = NestedPlates
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    # No observation type survives these all-empty domains. The scalar
    # endpoint's Float64 result declaration supplies the reduction identity.
    for data in ([(), ()], Tuple{}[]), name in (:object_observations, :method_observations)
        spec = getfield(N, name)
        reader = prepare(spec)
        grouped = prepare(spec; want=:grouped)(data, 0.1, 1.3)
        @test grouped == zeros(length(data))
        @test eltype(grouped) === Float64
        @test reader(data, 0.1, 1.3) === 0.0
        bound = prepare(spec; bound=(; groups=data))
        @test bound(0.1, 1.3) === 0.0
        for (k, args) in ((reader, (data, 0.1, 1.3)), (bound, (0.1, 1.3)))
            ad = prepare_ad(k, backend, args...; active=(:mu, :sigma))
            value, gradient = ad_value_and_gradient(ad, args...)
            @test value === 0.0
            @test all(iszero, gradient)
        end
        @test N.head_count(code_expr(reader), :for) == 2
    end
end
