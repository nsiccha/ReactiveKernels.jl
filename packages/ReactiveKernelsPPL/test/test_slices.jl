module SliceTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Random
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Globals, a unit effect nested in regions, a device effect crossing them and a
# hyperparameter (`sigma`) that no observation reads.
const MODEL = quote
    mu ~ Normal(0, 2)
    sigma ~ Exponential(1)
    tau ~ Exponential(1)
    beta ~ Normal(0, 1)
    s ~ Exponential(1)
    z[levels(unit)] .~ Normal.(0, sigma)
    d[levels(device)] .~ Normal.(0, tau)
    @plate for k in eachindex(y)
        y[k] ~ Normal(mu + z[unit[k]] + d[device[k]] + beta * x[k], s)
    end
end

# Unit labels chosen so that a region's units sit at other positions of the
# whole binding's `levels(unit)` than of the region's own binding.
const UNITS = Dict("north" => ["st-07", "st-02", "st-11"],
                   "south" => ["st-05", "st-09", "st-01", "st-12"],
                   "west"  => ["st-03", "st-08"])
const DEVICES = Dict("north" => ["dev-c", "dev-a"], "south" => ["dev-b", "dev-d"],
                     "west" => ["dev-a", "dev-d"])

function rows()
    rng = Xoshiro(7)
    unit, device, region, x, y = String[], String[], String[], Float64[], Float64[]
    for r in sort!(collect(keys(UNITS))), (j, st) in enumerate(UNITS[r]), t in 1:4
        push!(unit, st); push!(region, r)
        push!(device, DEVICES[r][mod1(j + t, 2)])
        push!(x, randn(rng)); push!(y, randn(rng) + 0.3)
    end
    (; unit, device, x, y), region
end

bind(model, data) = (b = bind_data(lower_rkppl(model, data; conditioned = (:y,),
    mod = @__MODULE__), data); (b, build_kernel(b)))

const DATA, REGION = rows()
const BOUND, BUILT = bind(MODEL, DATA)
const U = collect(range(-0.4, 0.5; length = BUILT.layout.total))

# The whole binding's packed positions of another binding's coordinates,
# matched by identity.
function positions_in(whole, part)
    index = Dict(id => i for (i, id) in enumerate(coordinate_identities(whole.layout)))
    [index[id] for id in coordinate_identities(part.layout)]
end

function own_binding(regions)
    keep = findall(in(regions), REGION)
    bound, built = bind(MODEL, map(v -> v[keep], DATA))
    at = positions_in(BUILT, built)
    sampler = prepare_sampler(built, bound, U[at]; backend = BACKEND)
    g = zeros(length(at))
    value, _ = sampler_value_and_gradient!(sampler, g, U[at])
    (; at, value, g)
end

@testset "coordinate identities name coordinates by their level labels" begin
    ids = coordinate_identities(BUILT.layout)
    names = coordinate_names(BUILT.layout)
    @test length(ids) == BUILT.layout.total
    @test allunique(ids)
    @test ids[findfirst(==(:mu), names)] == (path = (:mu,), index = ())
    z = findall(id -> id.path == (:z,), ids)
    @test [only(ids[i].index) for i in z] == sort(unique(DATA.unit))
    d = findall(id -> id.path == (:d,), ids)
    @test [only(ids[i].index) for i in d] == sort(unique(DATA.device))
    # A region's binding puts `st-05` at another position, under the same identity.
    south = findall(==("south"), REGION)
    _, part = bind(MODEL, map(v -> v[south], DATA))
    part_ids = coordinate_identities(part.layout)
    k = findfirst(==((path = (:z,), index = ("st-05",))), part_ids)
    @test coordinate_names(part.layout)[k] == Symbol("z.2")
    @test names[findfirst(==((path = (:z,), index = ("st-05",))), ids)] == Symbol("z.4")

    # Two axes, one labelled and one positional; `unique` keeps first occurrence.
    two = quote
        w[levels(g), 1:2] .~ Normal.(0, 1)
        v[unique(g)] .~ Normal.(0, 1)
        @plate for k in eachindex(y)
            y[k] ~ Normal(w[g[k], 1] + w[g[k], 2] + v[g[k]], 1)
        end
    end
    data = (; g = ["b", "a", "b"], y = [0.1, 0.2, 0.3])
    _, built = bind(two, data)
    ids = coordinate_identities(built.layout)
    @test [id.index for id in ids if id.path == (:w,)] == [("a", 1), ("b", 1), ("a", 2), ("b", 2)]
    @test [id.index for id in ids if id.path == (:v,)] == [("b",), ("a",)]
end

@testset "a region's slice of the whole binding equals the region's own binding" begin
    slices = prepare_slices(BUILT, BOUND, :y, U; by = (; y = REGION), backend = BACKEND)
    @test slice_labels(slices) == ["north", "south", "west"]
    for regions in (["south"], ["north"], ["west"], ["south", "west"])
        own = own_binding(regions)
        labels = length(regions) == 1 ? only(regions) : regions
        c = slice_coordinates(slices, labels)
        # Exactly the region's own coordinates, hyperparameters included.
        @test c == sort(own.at)
        @test findfirst(==(:sigma), coordinate_names(BUILT.layout)) in c
        g = zeros(length(c))
        value, _ = slice_value_and_gradient!(slices, g, U, labels)
        @test value ≈ own.value
        @test g[searchsortedfirst.(Ref(c), own.at)] ≈ own.g
        dense = fill(NaN, length(U))
        value2, _ = slice_value_and_gradient!(slices, dense, U, labels; compact = false)
        @test value2 == value
        @test dense[c] == g
        @test all(iszero, dense[setdiff(eachindex(U), c)])
        # Nothing outside the slice is read.
        moved = copy(U)
        moved[setdiff(eachindex(U), c)] .+= 3.0
        @test first(slice_value_and_gradient!(slices, zeros(length(c)), moved, labels)) == value
    end
    # A label set is a set: order and repetition do not matter.
    @test slice_coordinates(slices, ["west", "south", "west"]) ==
        slice_coordinates(slices, ("south", "west"))
    # Every label: the whole model, each prior counted once.
    whole = prepare_sampler(BUILT, BOUND, U; backend = BACKEND)
    g_whole = zeros(length(U))
    v_whole, _ = sampler_value_and_gradient!(whole, g_whole, U)
    all_labels = slice_labels(slices)
    @test slice_coordinates(slices, all_labels) == collect(eachindex(U))
    g = zeros(length(U))
    v, _ = slice_value_and_gradient!(slices, g, U, all_labels)
    @test v ≈ v_whole
    @test g ≈ g_whole
    @test_throws ContractValidationError slice_coordinates(slices, "east")
    @test_throws ContractValidationError slice_coordinates(slices, String[])
    @test_throws ContractValidationError slice_value_and_gradient!(slices, zeros(2), U, "south")
end

@testset "slices with shared values cached (partitioned)" begin
    region_of = Dict(u => r for (r, units) in UNITS for u in units)
    unit_regions = [region_of[u] for u in sort(unique(DATA.unit))]
    plain = prepare_slices(BUILT, BOUND, :y, U; by = (; y = REGION), backend = BACKEND)
    shared = prepare_slices(BUILT, BOUND, :y, U; by = (; y = REGION), backend = BACKEND,
        partitioned = (:z => unit_regions,))
    for labels in ("south", ["north", "west"], slice_labels(shared))
        c = slice_coordinates(plain, labels)
        @test slice_coordinates(shared, labels) == c
        a, ga = slice_value_and_gradient!(plain, zeros(length(c)), U, labels)
        b, gb = slice_value_and_gradient!(shared, zeros(length(c)), U, labels)
        @test b ≈ a
        @test gb ≈ ga
    end
end

@testset "a slice as a LogDensityProblems target" begin
    slices = prepare_slices(BUILT, BOUND, :y, U; by = (; y = REGION), backend = BACKEND)
    problem = slice_problem(slices, "west"; base = U)
    c = slice_coordinates(slices, "west")
    x = U[c] .+ 0.05
    @test LogDensityProblems.dimension(problem) == length(c)
    @test LogDensityProblems.capabilities(problem) == LogDensityProblems.LogDensityOrder{1}()
    point = copy(U); point[c] = x
    expected = slice_value_and_gradient!(slices, zeros(length(c)), point, "west")
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, x)
    @test value == first(expected)
    @test gradient == last(expected)
    @test LogDensityProblems.logdensity(problem, x) == value
    @test problem(x) == value
end

@testset "a density owning no coordinate is assigned only when the slice holds all it reads" begin
    coupled = quote
        sigma ~ Exponential(1)
        z[levels(unit)] .~ Normal.(0, sigma)
        total = sum(z)
        total ~ Normal(0, 0.5)
        @plate for k in eachindex(y)
            y[k] ~ Normal(z[unit[k]], 1)
        end
    end
    bound, built = bind(coupled, DATA)
    u = collect(range(-0.3, 0.3; length = built.layout.total))
    slices = prepare_slices(built, bound, :y, u; by = (; y = REGION), backend = BACKEND)
    @test_throws ContractValidationError slice_coordinates(slices, "south")
    # The whole model holds everything it reads.
    whole = prepare_sampler(built, bound, u; backend = BACKEND)
    g = zeros(length(u))
    v, _ = sampler_value_and_gradient!(whole, g, u)
    gs = zeros(length(u))
    vs, _ = slice_value_and_gradient!(slices, gs, u, slice_labels(slices))
    @test vs ≈ v
    @test gs ≈ g
end


# BRM emits a group's effects as rows of the positional twin `1:length(levels(g))`
# inside a block and gathers them by level position. Those rows are the levels of
# `g`, in `levels(g)` order, so their identities carry the level.
@rkppl twin_effects(g, scale) = begin
    z[1:length(levels(g))] .~ Normal.(0, scale)
    z
end

const TWIN = quote
    mu ~ Normal(0, 2)
    sigma ~ Exponential(1)
    s ~ Exponential(1)
    b ~ twin_effects(unit, sigma)
    @plate for k in eachindex(y)
        y[k] ~ Normal(mu + b[unit_position[k]], s)
    end
end

twin_data(keep) = (u = DATA.unit[keep];
    (; unit = u, unit_position = Int.(indexin(u, sort(unique(u)))), y = DATA.y[keep]))

@testset "a positional twin's rows are identified by their levels" begin
    whole_bound, whole = bind(TWIN, twin_data(eachindex(DATA.y)))
    ids = coordinate_identities(whole.layout)
    @test allunique(ids)
    @test [only(id.index) for id in ids if id.path == (:b, :z)] == sort(unique(DATA.unit))
    # `1:length(levels(g)) - k` does not say which levels it omits: positional.
    dropped = quote
        c[1:length(levels(g)) - 1] .~ Normal.(0, 1)
        y .~ Normal.(sum(c), 1)
    end
    _, built = bind(dropped, (; g = ["b", "a", "c"], y = [0.1, 0.2, 0.3]))
    @test [id.index for id in coordinate_identities(built.layout) if id.path == (:c,)] == [(1,), (2,)]

    u = collect(range(-0.3, 0.4; length = whole.layout.total))
    slices = prepare_slices(whole, whole_bound, :y, u; by = (; y = REGION), backend = BACKEND)
    for region in ("south", "north")
        keep = findall(==(region), REGION)
        own_bound, own = bind(TWIN, twin_data(keep))
        at = positions_in(whole, own)
        sampler = prepare_sampler(own, own_bound, u[at]; backend = BACKEND)
        g_own = zeros(length(at))
        v_own, _ = sampler_value_and_gradient!(sampler, g_own, u[at])
        c = slice_coordinates(slices, region)
        @test c == sort(at)
        g = zeros(length(c))
        v, _ = slice_value_and_gradient!(slices, g, u, region)
        @test v ≈ v_own
        @test g[searchsortedfirst.(Ref(c), at)] ≈ g_own
    end
end

end
