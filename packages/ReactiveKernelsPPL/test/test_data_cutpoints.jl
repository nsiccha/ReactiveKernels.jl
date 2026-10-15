using DifferentiationInterface, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# Independent cumulative-logit mathematics, including the stated scalar
# priors. No distribution adapter or derivative rule is used by this oracle.
_dc_logistic(x) = inv(1 + exp(-x))
function _dc_pointwise(u, x, y, cuts; discrimination = true, stopping = false)
    map(eachindex(y)) do i
        eta = u[1] + u[2] * x[i]
        d = discrimination ? exp(u[3] * x[i]) : 1.0
        k = y[i]
        if stopping
            lp = sum((log1p(-_dc_logistic(d * (cuts[j] - eta)))
                for j in 1:k-1); init = 0.0)
            return k == length(cuts) + 1 ? lp :
                lp + log(_dc_logistic(d * (cuts[k] - eta)))
        end
        hi = k == length(cuts) + 1 ? 1.0 : _dc_logistic(d * (cuts[k] - eta))
        lo = k == 1 ? 0.0 : _dc_logistic(d * (cuts[k-1] - eta))
        log(hi - lo)
    end
end
_dc_prior(u) = sum(x -> -0.5 * (log(2pi) + x^2), u)
_dc_sum(x::Real) = x
_dc_sum(xs) = sum(_dc_sum, xs; init = 0.0)
function _dc_fd(f, u)
    h = cbrt(eps(Float64))
    map(eachindex(u)) do i
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end
function _dc_bound(ast, data)
    plan = lower_rkppl(ast, data; mod = @__MODULE__, conditioned = keys(data))
    bind_data(plan, data)
end
function _dc_check(bound, u, pointwise)
    original = copy(u)
    built = build_kernel(bound)
    @test built.layout.total == length(u)
    @test isempty(bound.vector_parameters)
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    oracle(v) = _dc_prior(v) + _dc_sum(values(pointwise(v)))
    value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ oracle(u)
    @test grad ≈ _dc_fd(oracle, u) rtol = 2e-6 atol = 2e-8
    pw = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
    for name in keys(pointwise(u))
        @test getproperty(pw, name) ≈ getproperty(pointwise(u), name)
    end
    @test Base.invokelatest(prepare_query(built, bound, :prior), u) ≈ _dc_prior(u)
    @test Base.invokelatest(prepare_query(built, bound, :log_jacobian), u) == 0.0
    @test u == original
    return (; built, sampler)
end

function _dc_plate(cell; iterator = :(eachindex(y)))
    Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
        Expr(:for, :(i = $iterator), Expr(:block, cell)))
end
function _dc_rows(v, data; stopping = false)
    [ismissing(data.y[i]) ? 0.0 : only(_dc_pointwise(v, [data.x[i]], [data.y[i]], data.banks[data.g[i]];
        stopping)) for i in eachindex(data.y)]
end
_dc_nodes(ex) = ex isa Expr ? 1 + sum(_dc_nodes, ex.args; init = 0) : 1
_dc_loops(ex) = ex isa Expr ? Int(ex.head === :for) +
    sum(_dc_loops, ex.args; init = 0) : 0

@testset "indexed per-observation vectors retain plate structure" begin
    for stopping in (false, true), spelling in (:broadcast, :indexed, :indexed_ref)
        structure = stopping ? :StoppingRatio : :Cumulative
        ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            gamma ~ Normal(0, 1)
        end
        if spelling === :broadcast
            push!(ast.args, :(eta = a .+ b .* x),
                :(y .~ Ordinal.($structure(), LogitLink(), eta, banks[g], exp.(gamma .* x))))
        else
            cuts = spelling === :indexed_ref ? :(Ref(banks[g[i]])) : :(banks[g[i]])
            push!(ast.args, _dc_plate(:(y[i] ~ Ordinal($structure(), LogitLink(),
                a + b * x[i], $cuts, exp(gamma * x[i])))))
        end
        shapes = Tuple{Int,Int}[]
        for n in (6, 30)
            data = (; y = repeat([1, 4, 2, 1, 3, 1], n ÷ 6),
                x = collect(range(-0.6, 0.8; length = n)),
                g = repeat([1, 2, 1, 3, 2, 3], n ÷ 6),
                banks = [[-0.7, 1.1], [-1.4, -0.2, 0.9], Float64[]])
            result = _dc_check(_dc_bound(ast, data), [0.2, -0.3, 0.15],
                v -> (; y = _dc_rows(v, data; stopping)))
            source = Base.remove_linenums!(deepcopy(code_expr(result.sampler.kernel)))
            push!(shapes, (_dc_nodes(source), _dc_loops(source)))
            @test last(shapes)[2] > 0
            if n == 6
                bad = merge(data, (; y = [1, 4, 2, 2, 3, 1]))
                @test_throws ContractValidationError _dc_bound(ast, bad)
                if spelling !== :broadcast
                    # A literal iterator covers the same whole response
                    # as eachindex, preserving the indexed data meaning.
                    ranged = deepcopy(ast)
                    ranged.args[end].args[end].args[1] = :(i = 1:6)
                    _dc_check(_dc_bound(ranged, data), [0.2, -0.3, 0.15],
                        v -> (; y = _dc_rows(v, data; stopping)))
                    # The absent observation must not index banks[99].
                    masked = merge(data, (; y = Union{Missing,Int}[1, missing, 2, 1, 3, 1],
                        g = [1, 99, 1, 3, 2, 3]))
                    _dc_check(_dc_bound(ast, masked), [0.2, -0.3, 0.15],
                        v -> (; y = _dc_rows(v, masked; stopping)))
                end
            end
        end
        @test shapes[1] == shapes[2]
    end
end

@testset "shared ordinal data cutpoints keep data semantics" begin
    for kind in (:ordered, :cumulative, :stopping)
        response = kind === :ordered ? :(OrderedLogistic.(eta, Ref(cuts))) :
            :(Ordinal.($(kind === :cumulative ? :Cumulative : :StoppingRatio)(),
                LogitLink(), eta, Ref(cuts), exp.(gamma .* x)))
        ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
        end
        kind === :ordered || push!(ast.args, :(gamma ~ Normal(0, 1)))
        push!(ast.args, :(y .~ $response))
        data = (; y = [1, 3, 2, 1, 2, 3],
            x = [-0.6, 0.2, 0.8, -0.1, 0.4, 0.7], cuts = [-0.7, 1.1])
        saved = deepcopy(data)
        bound = _dc_bound(ast, data)
        u = kind === :ordered ? [0.2, -0.3] : [0.2, -0.3, 0.15]
        oracle(v) = (; y = _dc_pointwise(v, data.x, data.y, data.cuts;
            discrimination = kind !== :ordered, stopping = kind === :stopping))
        _dc_check(bound, u, oracle)
        @test data == saved

        empty = merge(data, (; x = Float64[], y = Int[]))
        _dc_check(bind_data(bound, empty), u, v -> (; y = Float64[]))

        # Rebinding changes both the support and the selected data values.
        changed = merge(data, (; cuts = [-1.3, -0.1, 0.8, 1.6],
            y = [1, 5, 2, 4, 3, 5]))
        rebound = bind_data(bound, changed)
        @test only(rebound.responses).n_levels == 5
        _dc_check(rebound, u, v -> (; y = _dc_pointwise(v, changed.x,
            changed.y, changed.cuts; discrimination = kind !== :ordered,
            stopping = kind === :stopping)))

        # Refused: the law needs one real threshold vector and integer
        # outcomes within that vector's support (mathematical domain).
        @test_throws ContractValidationError bind_data(bound,
            merge(data, (; cuts = ones(2, 2))))
        @test_throws ContractValidationError bind_data(bound,
            merge(data, (; cuts = ["low", "high"])))
        @test_throws ContractValidationError bind_data(bound,
            merge(data, (; y = [1, 4, 2, 1, 2, 3])))
        if kind !== :stopping
            bad = bind_data(bound, merge(data, (; cuts = [1.1, -0.7])))
            q = prepare_query(build_kernel(bad), bad, :likelihood)
            @test Base.invokelatest(q, u) == -Inf
        end
    end
end

@testset "indexed shared cutpoints and aliases bind independently" begin
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        gamma ~ Normal(0, 1)
        second = banks[right]
        eta1 = a .+ b .* x1
        eta2 = a .+ b .* x2
        y1 .~ Ordinal.(Cumulative(), LogitLink(), eta1, Ref(banks[left]), exp.(gamma .* x1))
        y2 .~ Ordinal.(Cumulative(), LogitLink(), eta2, Ref(second), exp.(gamma .* x2))
    end
    data = (; x1 = [-0.5, 0.1, 0.7], x2 = [0.3, -0.1, 0.4, 0.9],
        y1 = [1, 2, 3], y2 = [2, 4, 1, 3],
        banks = [[-0.9, 0.6], [-1.3, 0.2, 1.4]], left = 1, right = 2)
    bound = _dc_bound(ast, data)
    u = [0.2, -0.3, 0.15]
    _dc_check(bound, u, v -> (; y1 = _dc_pointwise(v, data.x1, data.y1, data.banks[1]),
        y2 = _dc_pointwise(v, data.x2, data.y2, data.banks[2])))
    changed = merge(data, (; left = 2, right = 1,
        banks = [[-0.5, 0.4], [-1.0, 0.2, 0.7]],
        y1 = [1, 2, 4], y2 = [2, 3, 1, 2]))
    _dc_check(bind_data(bound, changed), u,
        v -> (; y1 = _dc_pointwise(v, changed.x1, changed.y1, changed.banks[2]),
            y2 = _dc_pointwise(v, changed.x2, changed.y2, changed.banks[1])))
    # Refused: Julia indexing must be in bounds; binding cannot silently
    # select another group's thresholds.
    @test_throws ContractValidationError _dc_bound(ast, merge(data, (; left = 3)))
end

@testset "group-indexed Ref shares each group's own cutpoints" begin
    for structure in (:Cumulative, :StoppingRatio)
        ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            gamma ~ Normal(0, 1)
        end
        push!(ast.args, _dc_plate(:(y[i] .~ Ordinal.($structure(), LogitLink(),
            a .+ b .* x[i], Ref(banks[i]), exp.(gamma .* x[i])))))
        data = (; y = [[1, 3, 2], [4, 2], [1]],
            x = [[-0.3, 0.7, 0.2], [0.4, -0.6], [0.9]],
            banks = [[-0.8, 1.2], [-1.1, 0.1, 1.6], Float64[]])
        oracle(v) = (; y = [_dc_pointwise(v, data.x[i], data.y[i], data.banks[i];
            stopping = structure === :StoppingRatio) for i in eachindex(data.y)])
        _dc_check(_dc_bound(ast, data), [0.2, -0.3, 0.15], oracle)
    end
end
