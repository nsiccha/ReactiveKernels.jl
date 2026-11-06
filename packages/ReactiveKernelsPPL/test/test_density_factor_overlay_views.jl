module DensityFactorOverlayViewTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme

const _REVERSE = ReactiveKernels.DifferentiationInterface.AutoEnzyme(; mode = Enzyme.Reverse)

# A compact factor query reads the packed point through `_CompactCellPoint`, an
# overlay of the constant base and the active vector. A plate over a block read
# as `view(u, r)` restricts it to the selected rows with `view(_, rows::Vector{Int})`,
# whose `unalias` reaches Base's default `dataids`, `objectid(A)`, so the overlay's
# stack slot goes to an opaque runtime call. In a body large enough that the tuple
# of constant operands stays on the stack, the overlay's base pointer is then a
# load from that tuple, and native Reverse refuses to store it into the overlay
# (`EnzymeRuntimeActivityError`) although it only ever holds a `Const` argument.
# The derivative is defined: the compact form agrees with the full one wherever
# Enzyme accepts it. The repair belongs in Enzyme (snag `static-reverse-o-c5bf409d`),
# so the compact gradient here is either exact or this refusal, recorded as broken.
# The law is a generic space-time precision; only its graph's size matters here.
struct OverlayViewField{S,T,L,G}
    spatial_length::S
    temporal_length::T
    correlation_factor::L
    geometry::G
end

function _sparse_slot(rows, pointers, column, row)
    first, last = pointers[column], pointers[column + 1] - 1
    while first <= last
        middle = first + (last - first) ÷ 2
        candidate = rows[middle]
        candidate == row && return middle
        if candidate < row
            first = middle + 1
        else
            last = middle - 1
        end
    end
    0
end

ReactiveKernels.@kernel function _factor_graph(mass_scale, stiffness_scale, geometry)
    factor = scan(eachindex(geometry.lrow); init = 0, history = 0.0) do carry, p, earlier
        i = geometry.lrow[p]
        j = geometry.columns[p]
        corrections = plate(geometry.previous_ptr[j]:(geometry.previous_ptr[j + 1] - 1)) do q
            k = geometry.previous_col[q]
            slot = _sparse_slot(geometry.lrow, geometry.lptr, k, i)
            value = slot == 0 ? 0.0 : earlier[slot] *
                earlier[geometry.previous_pos[q]] * earlier[geometry.lptr[k]]
            return value
        end
        raw = stiffness_scale * geometry.stiffness[p] +
            (i == j ? mass_scale * geometry.mass[j] : 0.0)
        numerator = raw - sum(corrections)
        entry = i == j ? numerator : numerator / earlier[geometry.lptr[j]]
        (carry, entry)
    end
    return factor
end

ReactiveKernels.@kernel function _field_density(field, spatial_length, temporal_length, L, geometry)
    N = length(geometry.mass)
    C = geometry.nchannels
    T = length(geometry.times)
    factor = _factor_graph(1.0 / spatial_length, spatial_length / 2.0, geometry)
    logdet_space = 2.0 * sum(log.(factor[geometry.lptr[1:N]])) -
        sum(log.(geometry.mass)) - N * log(2.0 * pi)
    channel_logs = plate(1:C) do c
        return log(L[c, c])
    end
    normalizer = T * C * logdet_space / 2.0 -
        N * C * T * log(2.0 * pi) / 2.0 - N * T * sum(channel_logs)
    transitions = plate(1:T) do t
        gap = t == 1 ? 0.0 : geometry.times[t] - geometry.times[t - 1]
        decay = t == 1 ? 0.0 : exp(-gap / temporal_length)
        variance = t == 1 ? 1.0 : -expm1(-2.0 * gap / temporal_length)
        vertices = plate(1:N) do i
            residual = scan(1:C; init = 0, history = 0.0) do carry, c, earlier
                offset = N * ((c - 1) + C * (t - 1))
                index = geometry.permutation[i] + offset
                innovation = field[index] - (t == 1 ? 0.0 : decay * field[index - N * C])
                neighbours = plate(geometry.gptr[i]:(geometry.gptr[i + 1] - 1)) do p
                    neighbour = geometry.permutation[geometry.grow[p]] + offset
                    difference = field[neighbour] -
                        (t == 1 ? 0.0 : decay * field[neighbour - N * C])
                    return geometry.gval[p] * difference
                end
                spatial = geometry.mass[i] * (innovation / spatial_length) +
                    (spatial_length / 2.0) * sum(neighbours)
                previous = sum((earlier[k] * L[c, k] for k in 1:(c - 1)); init = 0.0)
                white = (spatial - previous) / L[c, c]
                (carry, white)
            end
            scale = sqrt(4.0 * pi * geometry.mass[i]) * sqrt(variance)
            return sum(abs2, residual ./ scale)
        end
        return sum(vertices) + N * C * log(variance) / 2.0
    end
    return normalizer - sum(transitions)
end

ReactiveKernels.@kernel function _field_coordinates(u, shape, spatial_length, temporal_length, L, geometry)
    return u
end
ReactiveKernels.@kernel function _field_logjac(u, shape, spatial_length, temporal_length, L, geometry)
    return 0.0
end

function ReactiveKernelsPPL.sampling_geometry(::Type{OverlayViewField}, args, shape)
    dimension = :($(args[4]).ncoordinates)
    ParameterGeometry(isempty(shape) ? (dimension,) : shape, dimension; support = :real,
        constrain = _field_coordinates, unconstrain = _field_coordinates, logjac = _field_logjac)
end
ReactiveKernelsPPL.sampling_distribution_graph(::Type{OverlayViewField}, args::Tuple) =
    _field_density

# `N` unconnected vertices at `T` times: a diagonal factor and no fill-in.
_toy_geometry(N, T) = (; permutation = collect(1:N), mass = ones(N), gptr = collect(1:N+1),
    grow = collect(1:N), gval = ones(N), lptr = collect(1:N+1), lrow = collect(1:N),
    previous_ptr = ones(Int, N + 1), previous_col = Int[], previous_pos = Int[],
    columns = collect(1:N), stiffness = ones(N), times = collect(0.0:T-1), nchannels = 1,
    ncoordinates = N * T)

@testset "a compact factor query beside a plate read through a view" begin
    model = @rkppl begin
        mesh = _toy_geometry(N, T)
        spatial_length ~ LogNormal(log(0.5), 0.5)
        temporal_length ~ LogNormal(log(10.0), 0.5)
        field[1:mesh.ncoordinates] ~ OverlayViewField(spatial_length, temporal_length, L, mesh)
        tau[1:1] .~ Normal.(0.0, 1.0)
        z[1:4] .~ Normal.(0.0, 1.0)
    end
    plan = bind_data(model(; N = 4, T = 4, L = [1.0;;]))
    built = build_kernel(plan)
    u = 0.1 .* sin.(1:built.layout.total)
    original = copy(u)
    catalogue = density_factors(built, plan)
    selection = select_factors(catalogue, collect(catalogue))
    value = prepare_query(built, plan, selection; u0 = u)

    # The full derivative of the same factors, against central differences.
    full = only(prepare_query_ad(prepare_ad, built, plan, [selection], _REVERSE, u))
    full_value, full_gradient = ad_value_and_gradient(full, u)
    @test full_value ≈ value(u)
    h = 1e-5
    central = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h
        minus[j] -= h
        (value(plus) - value(minus)) / (2h)
    end
    @test full_gradient ≈ central rtol = 1e-6 atol = 1e-8

    q = only(prepare_query_ad(prepare_ad, built, plan, [selection], _REVERSE, u;
        coordinates = :support))
    @test q.coordinates == factor_support(selection; u0 = u)
    outcome = try
        ad_value_and_gradient(q, u, u[q.coordinates])
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        err
    end
    if outcome isa Exception
        @test occursin("Mismatched activity", sprint(showerror, outcome))
        @test_broken false
    else
        compact_value, compact_gradient = outcome
        @test compact_value ≈ full_value
        @test compact_gradient ≈ full_gradient[q.coordinates]
        @test all(iszero, full_gradient[setdiff(eachindex(u), q.coordinates)])
    end
    @test u == original
end

end
