using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jhcol_search(theta::Float64, flags::Vector{Bool}, x::Vector{Float64}) = begin
    yes = findall(flags)
    no = findall(!, flags)
    positive = findall(>(0.0), x)
    result = theta * (sum(yes) + 2sum(no) + 3sum(positive))
    return result
end
@kernel jhcol_lookup(theta::Float64, labels::Vector{Int}, payload::Vector{Float64}, queries::Vector{Int}) = begin
    lookup = Dict(zip(labels, payload))
    cells = plate(queries, Ref(lookup)) do key, d
        haskey(d, key) ? d[key] : get(d, key, -2.0)
    end
    result = theta * sum(cells)
    return result
end
@traceable function jhcol_reuse_zip(labels, payload)
    z = zip(labels, payload)
    first = Dict(z)
    second = Dict{Float64,Float64}(z)
    result = 0.0
    for (key, value) in z
        result += first[key] + second[key] + value
    end
    result
end
@kernel jhcol_reuse(theta::Float64, labels::Vector{Int}, payload::Vector{Int}) = begin
    coefficient = jhcol_reuse_zip(labels, payload)
    result = theta * coefficient
    return result
end
@traceable function jhcol_zero_zip(n)
    count = 0
    for pair in zip()
        count += 1
        count >= n && break
    end
    z = zip()
    for pair in z
        count += 1
        count >= 2n && break
    end
    count
end
@kernel jhcol_zero_zip_kernel(theta::Float64, n::Int) = begin
    count = jhcol_zero_zip(n)
    result = theta * count
    return result
end
@traceable function jhcol_float_dict(keys, values, queries)
    d = Dict(zip(keys, values))
    result = 0.0
    for k in queries
        result += get(d, k, -3.0)
    end
    result + length(d)
end
@kernel jhcol_floats(theta::Float64, keys::Vector{Float64}, values::Vector{Float64}, queries::Vector{Float64}) = begin
    coefficient = jhcol_float_dict(keys, values, queries)
    result = theta * coefficient
    return result
end
@kernel jhcol_ragged(theta::Float64, rows::Vector{Vector{Float64}}) = begin
    positions = plate(rows) do row
        findall(x -> x > 1.0, row)
    end
    counts = plate(positions) do row
        sum(row)
    end
    result = theta * sum(counts)
    return result
end

# A binder's definitions call kernels whose plans contain the searches and
# dictionaries. Preparation must derive every input from the raw columns.
@kernel jhcol_selected(flags::Vector{Bool}) = begin
    rows = findall(flags)
    return rows
end
@kernel jhcol_rejected(flags::Vector{Bool}) = begin
    rows = findall(!, flags)
    return rows
end
@kernel jhcol_geometry(values::Vector{Float64}, selected::Vector{Int}, rejected::Vector{Int}) = begin
    n = length(selected) + length(rejected)
    rows = 1:n
    known = Dict(zip(selected, values))
    positions = Dict(zip(rejected, eachindex(rejected)))
    base = plate(rows) do row
        get(known, row, 0.0)
    end
    order = plate(rows) do row
        haskey(positions, row) ? n + positions[row] : row
    end
    return (base, order)
end
@kernel jhcol_defined(theta::Float64, base::Vector{Float64}, order::Vector{Int}) = begin
    result = theta * sum(base) + theta^2 * sum(order)
    return result
end

function jhcol_check(spec, have, datasets)
    source = jax_source(spec; have, want = :result, data = first(datasets))
    native = prepare(spec; have, want = :result)
    @test occursin("def prepare_data", string(source))
    for data in datasets
        @test string(jax_source(spec; have, want = :result, data)) == string(source)
        if rk_jax_available()
            args = (0.7, values(data)...)
            unchanged = deepcopy(args)
            got = rk_jax_evaluate(source, :kernel, args; grad = true, prepare = :arguments)
            @test got.value ≈ native(args...)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(first(args)), map(Enzyme.Const, Base.tail(args))...)[1][1]
            @test only(got.grad) ≈ derivative
            @test isequal(args, unchanged)
        end
    end
    rk_jax_available() || @test_skip false
    source
end

@testset "jax_source standalone collection definitions" begin
    have = (:theta, :base, :order)
    definitions = [:selected => :($jhcol_selected(flags)),
                   :rejected => :($jhcol_rejected(flags)),
                   :geometry => :($jhcol_geometry(values, selected, rejected)),
                   :base => :(geometry[1]), :order => :(geometry[2])]
    datasets = [(; flags = [false, true, false, true], values = [2.0, 3.0]),
                (; flags = Bool[], values = Float64[]),
                (; flags = [false, false, false], values = Float64[]),
                (; flags = [true, true], values = [1.0, 7.0])]
    function data_for(raw)
        selected, rejected = jhcol_selected(raw.flags), jhcol_rejected(raw.flags)
        geometry = jhcol_geometry(raw.values, selected, rejected)
        (; raw..., selected, rejected, geometry, base = geometry[1], order = geometry[2])
    end
    source = jax_source(jhcol_defined; have, want = :result,
                        data = data_for(first(datasets)), definitions)
    @test occursin("UNTRANSLATED = {}", string(source))
    names = first.(only(f for f in source.functions if f.name == "prepare_data").inputs)
    @test Set(names) == Set(["flags", "values"])
    native = prepare(jhcol_defined; have, want = :result)
    for raw in datasets
        data = data_for(raw)
        @test string(jax_source(jhcol_defined; have, want = :result, data, definitions)) == string(source)
        if rk_jax_available()
            args = Tuple(getproperty(raw, Symbol(name)) for name in names)
            got = rk_jax_evaluate(source, :kernel, (0.7, args...); grad = true, prepare = :arguments)
            @test got.value ≈ native(0.7, data.base, data.order)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(0.7), Enzyme.Const(data.base), Enzyme.Const(data.order))[1][1]
            @test only(got.grad) ≈ derivative
        end
    end
    rk_jax_available() || @test_skip false
end

@testset "jax_source host searches and zipped dictionaries" begin
    jhcol_check(jhcol_search, (:theta, :flags, :x),
        [(; flags = [true, false, true], x = [-1.0, 0.0, 2.0]),
         (; flags = Bool[], x = Float64[]),
         (; flags = falses(7) |> collect, x = ones(7)),
         (; flags = trues(20) |> collect, x = fill(-2.0, 20))])
    if rk_jax_available()
        data = (; flags = [false, true, false, true], x = [1.0, -2.0, 3.0, 0.0])
        details = jax_source(jhcol_search; have = (:theta, :flags, :x),
            functions = (kernel = :result, positions = (:yes, :no, :positive)), data)
        got = rk_jax_evaluate(details, :positions, (0.7, values(data)...); prepare = true)
        @test got.value == [[2, 4], [1, 3], [1, 3]]
    end
    source = jhcol_check(jhcol_lookup, (:theta, :labels, :payload, :queries),
        [(; labels = [2, 4, 2], payload = [1.5, 3.0, 7.0], queries = [4, 3, 2]),
         (; labels = Int[], payload = Float64[], queries = [1, 2]),
         (; labels = [1, 2, 3], payload = [5.0], queries = [1, 2, 3]),
         (; labels = [1, 2], payload = [5.0, 6.0, 7.0], queries = Int[])])
    @test occursin("zip", string(source))
    @test occursin(r"\bfor\b", string(source))
    jhcol_check(jhcol_reuse, (:theta, :labels, :payload),
        [(; labels = [2, 1, 2], payload = [4, 5, 6]),
         (; labels = Int[], payload = Int[])])
    jhcol_check(jhcol_zero_zip_kernel, (:theta, :n), [(; n = 1), (; n = 7)])
    jhcol_check(jhcol_floats, (:theta, :keys, :values, :queries),
        [(; keys = [NaN, -0.0, 0.0, NaN], values = [1.0, 2.0, 3.0, 4.0],
            queries = [NaN, -0.0, 0.0, 7.0]),
         (; keys = Float64[], values = Float64[], queries = [1.0])])
    jhcol_check(jhcol_ragged, (:theta, :rows),
        [(; rows = [[0.0, 2.0], Float64[], [3.0]]),
         (; rows = [Float64[], [0.0, -1.0]]),
         (; rows = [[2.0, 3.0, 4.0, 5.0], [1.0], [9.0]])])
end
