# Independent synthetic array cells: ordinary callable leaves, no statistical
# library or downstream model implementation.
_acr_row(x, i) = x[i, :]

module ArrayCellRowsSubmodel
using ReactiveKernelsPPL
row(x, i) = x[i, :]
@rkppl rows(x, a, b) = begin
    @plate for i in axes(x, 1)
        mu = a .+ b .* row(x, i)
        location[i, 1:2] = mu
    end
    return location
end
end

function _acr_ast(kind)
    cell = kind === :literal ? :(row = [a, b]) :
        kind === :direct ? :(row = a .+ b .* x[i, :]) :
        :(row = a .+ b .* _acr_row(x, i))
    reads = kind === :columns ? :(reads = location[:, 1] .+ location[:, 2]) :
        kind === :alias ? quote
            alias = location
            reads = vec(permutedims(alias))
        end : kind === :matrix ? :(reads = location) :
        :(reads = vec(permutedims(location)))
    collected = kind === :submodel ?
        :(location ~ ArrayCellRowsSubmodel.rows(x, a, b)) : quote
            @plate for i in axes(x, 1)
                $cell
                location[i, 1:2] = row
            end
        end
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        $collected
        $reads
        y .~ Normal.(reads, sigma)
    end
    # Splice multiple ordinary statements into the model's outer block.
    ast.args = reduce(vcat, (Meta.isexpr(st, :block) ? st.args : Any[st]
        for st in ast.args); init=Any[])
    return ast
end

function _acr_build(kind, n)
    x = reshape([0.3sin(i) for i in 1:2n], n, 2)
    y = kind === :columns ? [0.2cos(i) for i in 1:n] :
        kind === :matrix ? reshape([0.2cos(i) for i in 1:2n], n, 2) :
        [0.2cos(i) for i in 1:2n]
    data = (; x, y)
    plan = lower_rkppl(_acr_ast(kind), (:x, :y);
        mod=@__MODULE__, conditioned=(:y,))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(data)))
    return (; built=build_kernel(bound), bound, data, u=[0.2, -0.3, 0.1])
end

function _acr_expected(kind, data, u)
    a, b, logsigma = u
    sigma = exp(logsigma)
    location = kind === :literal ? repeat([a b], size(data.x, 1), 1) :
        a .+ b .* data.x
    reads = kind === :columns ? vec(sum(location; dims=2)) :
        kind === :matrix ? location : vec(permutedims(location))
    pointwise = logpdf.(Normal.(reads, sigma), data.y)
    value = logpdf(Normal(), a) + logpdf(Normal(), b) +
        logpdf(Exponential(), sigma) + logsigma + sum(pointwise)
    return (; location, reads, pointwise, value)
end

function _acr_findiff(f, u)
    h = cbrt(eps(Float64))
    return [(f(u + h * e) - f(u - h * e)) / (2h)
        for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

function _acr_query(fx, node)
    names = sort!(collect(keys(fx.bound.columns)))
    values = NamedTuple{Tuple(names)}(Tuple(fx.bound.columns[k] for k in names))
    return Base.invokelatest(prepare, fx.built.spec;
        have=(:unconstrained, names...), want=node, bound=values)
end
