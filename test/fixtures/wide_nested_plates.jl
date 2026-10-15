module WideNestedPlates
using ReactiveKernels

# Width is authored graph structure, not a data length. Each operand is read
# inside the outer cell so the nested body's type queries must infer it there.
function kernel(width; explicit_refs=true)
    ports = [Symbol(:xs, k) for k in 1:(width - 2)]
    cells = [Symbol(:x, k) for k in 1:(width - 2)]
    reads = [:($p = data[$k]) for (k, p) in enumerate(ports)]
    total = foldl((a, b) -> :($a + $b), cells)
    inner = explicit_refs ? quote
        plate($(ports...), Ref(scale), Ref(weights)) do $(cells...), s, w
            s * $total + s^2 * w[2]
        end
    end : quote
        plate($(ports...)) do $(cells...)
            scale * $total + scale^2 * weights[2]
        end
    end
    Core.eval(@__MODULE__, quote
        @kernel function $(Symbol(:wide_nested_, width, :_, explicit_refs))(groups, scale, weights)
            rows = plate(groups) do data
                $(reads...)
                values = $inner
                sum(values)
            end
            return sum(rows)
        end
    end)
end

data(width, lengths) = [[sin.(k .+ (1:n)) for k in 1:(width - 2)] for n in lengths]

function reference(groups, scale, weights)
    # Independent array algebra also covers singleton-expanded matrix axes.
    totals = [reduce((x, y) -> x .+ y, d) for d in groups]
    linear = sum(x -> sum(x; init=0.0), totals; init=0.0)
    lanes = sum(length, totals; init=0)
    (scale * linear + scale^2 * weights[2] * lanes,
     linear + 2scale * weights[2] * lanes)
end

count_head(ex, head) = ex isa Expr ? Int(ex.head === head) +
    sum(x -> count_head(x, head), ex.args; init=0) : 0
end
