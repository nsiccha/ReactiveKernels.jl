module ComputedGatherModels
const calls = Ref(0)
function positions(labels)
    calls[] += 1
    return Int.(labels)
end
const aliased_positions = positions
labels(g) = string.(g)
passthrough(z) = z
end

function _cgi_source(kind)
    dims = kind === :second ? :(z[1:1, 1:length(levels(g))]) :
        kind === :levels ? :(z[levels(h), 1:1]) :
        :(z[1:length(levels(g)), 1:1])
    index = kind === :levels ? :(index = labels(g)) :
        kind === :alias ? :(index = aliased_positions(g)) : :(index = positions(g))
    read = kind === :second ? :(draws[1, index]) : :(draws[index, 1])
    definition = kind === :opaque ? :(draws = passthrough(z) .* scale) : :(draws = z .* scale)
    if kind in (:inline, :inline_response)
        index = nothing
        read = :(draws[positions(g), 1])
    elseif kind === :data_expr
        index = :(index = g .+ 0)
    end
    result = kind === :split ? quote
        selected = $read
        result = selected .* x
    end : :(result = $read .* x)
    response = kind === :inline_response ? :(y .~ Normal.($read .* x, 1)) :
        :(y .~ Normal.(result, 1))
    source = quote
        scale ~ Exponential(1)
        $dims .~ Normal.(0, 1)
        $definition
        $index
        $(kind === :inline_response ? nothing : result)
        $response
    end
    flat = Any[]
    for statement in source.args
        if Meta.isexpr(statement, :block)
            append!(flat, statement.args)
        elseif statement !== nothing
            push!(flat, statement)
        end
    end
    return Expr(:block, flat...)
end

function _cgi_build(kind, G, n)
    g = [mod1(3i + div(i, max(G, 1)), G) for i in 1:n]
    # Cover the entire declared positional axis, preserving repeats and order.
    n >= G && G > 0 && (g[1:G] = reverse(1:G))
    data = Dict{Symbol,Any}(:g => g, :x => [sin(0.4i) for i in 1:n],
        :y => [cos(0.7i) / 3 for i in 1:n])
    kind === :levels && (data[:h] = string.(g))
    source = _cgi_source(kind)
    unbound = lower_rkppl(source, Tuple(keys(data));
        mod=ComputedGatherModels, conditioned=(:y,))
    before = ComputedGatherModels.calls[]
    bound = bind_data(unbound, data)
    calls = ComputedGatherModels.calls[] - before
    built = build_kernel(bound)
    u = [0.2sin(0.8i) for i in 1:built.layout.total]
    return (; data, source, unbound, bound, built, u, kind, calls)
end

function _cgi_oracle(fx, u)
    nt = constrain(fx.built.layout, u)
    draws = nt.z .* nt.scale
    index = fx.kind === :levels ?
        Int[findfirst(==(string(i)), sort(unique(fx.data[:h]))) for i in fx.data[:g]] : fx.data[:g]
    selected = fx.kind === :second ? draws[1, index] : draws[index, 1]
    location = selected .* fx.data[:x]
    pointwise = logpdf.(Normal.(location, 1), fx.data[:y])
    prior = logpdf(Exponential(1), nt.scale) + sum(logpdf.(Normal(), nt.z))
    jacobian = log(nt.scale)
    return (; draws, location, pointwise, prior, jacobian,
        value=sum(pointwise) + prior + jacobian)
end

function _cgi_query(fx, node)
    names = sort!(collect(keys(fx.bound.columns)))
    values = NamedTuple{Tuple(names)}(Tuple(fx.bound.columns[k] for k in names))
    return Base.invokelatest(prepare, fx.built.spec;
        have=(:unconstrained, names...), want=node, bound=values)
end

function _cgi_findiff(f, u; h=1e-6)
    map(eachindex(u)) do i
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end
