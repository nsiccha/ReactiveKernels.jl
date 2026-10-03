module TensorizedCatReactantTests

using ReactiveKernels, Reactant, Enzyme, Test

const join_columns = hcat
const join_rows = vcat
const join_axis = cat

module Custom
    hcat(a, x) = a + sum(x)
end

# Resolved call heads occur in generated programs, including PPL programs.
# These independent synthetic kernels exercise the core compiler boundary.
function cat_kernel(callee; dims = nothing, vector_head = false)
    name = gensym(:resolved_cat)
    call = dims === nothing ? Expr(:call, callee, :a, :x) :
        Expr(:call, callee, Expr(:parameters, Expr(:kw, :dims, dims)), :a, :x)
    head = vector_head ? :(u[1] .* x) : :(sum(u[1:1]))
    prediction = vector_head ? :(X * u[2:3]) : :(sum(vec(X) .* u[2:3]))
    loss = vector_head ? :(sum(abs2, mu) + sum(abs2, u)) :
        :(mu^2 + sum(abs2, u))
    def = Expr(:(=), Expr(:call, name, :u, :x), quote
        a = $head
        X = $call
        mu = $prediction
        loss = $loss
        return loss
    end)
    Core.eval(@__MODULE__, :(@kernel $def))
    getglobal(@__MODULE__, name)
end

gradient(f, u) = only(Enzyme.gradient(Enzyme.Reverse, f, u))
operation_names(hlo) = [m.match for m in eachmatch(
    r"\b(?:stablehlo|chlo|enzyme|func|arith|scf|cf|tensor|math|linalg|memref)\.\w+", repr(hlo))]

@testset "resolved cat calls keep ordinary scalar concatenation" begin
    u, x = [0.2, 0.1, 0.3], [0.43]
    ru = Reactant.to_rarray(u)
    for (callee, dims) in ((:hcat, nothing), (:(Base.hcat), nothing),
            (GlobalRef(Base, :hcat), nothing), (GlobalRef(@__MODULE__, :hcat), nothing),
            (:join_columns, nothing), (GlobalRef(@__MODULE__, :join_columns), nothing))
        spec = cat_kernel(callee; dims)
        kernel = prepare(spec; bound = (; x))
        matrix = prepare(spec; bound = (; x), want = :X)
        @test size(matrix(u)) == (1, 2)
        @test Array((Reactant.@compile matrix(ru))(ru)) == hcat(u[1], x)
        # refused: Julia concatenation requires matching row counts;
        # a scalar is one row and does not broadcast to fill a column.
        @test_throws DimensionMismatch prepare(spec)(u, [0.43, 0.71])
        primal = Reactant.@compile kernel(ru)
        reverse(v) = gradient(kernel, v)
        compiled_reverse = Reactant.@compile reverse(ru)
        for shift in (0.0, 0.17)
            v = u .+ shift
            rv = Reactant.to_rarray(v)
            mu = v[1] * v[2] + x[1] * v[3]
            value = mu^2 + sum(abs2, v)
            expected = 2v .+ 2mu .* [v[2], v[1], x[1]]
            @test kernel(v) ≈ value
            @test gradient(kernel, v) ≈ expected
            @test Float64(primal(rv)) ≈ value
            @test Array(compiled_reverse(rv)) ≈ expected
            @test Array(rv) == v
        end
        @test x == [0.43]
    end
end

@testset "resolved array concatenation reuses the existing wrappers" begin
    A, B = [0.2 0.4], [0.3 0.5]
    for (callee, dims, ordinary) in ((GlobalRef(Base, :hcat), nothing, hcat),
            (GlobalRef(Base, :vcat), nothing, vcat), (:join_rows, nothing, vcat),
            (GlobalRef(Base, :cat), 2, (a, b) -> cat(a, b; dims = 2)),
            (:join_axis, 2, (a, b) -> cat(a, b; dims = 2)))
        name = gensym(:array_cat)
        call = dims === nothing ? Expr(:call, callee, :A, :B) :
            Expr(:call, callee, Expr(:parameters, Expr(:kw, :dims, dims)), :A, :B)
        def = Expr(:(=), Expr(:call, name, :A, :B), quote
            X = $call
            return X
        end)
        Core.eval(@__MODULE__, :(@kernel $def))
        kernel = prepare(getglobal(@__MODULE__, name); bound = (; B))
        ra = Reactant.to_rarray(A)
        compiled = Reactant.@compile kernel(ra)
        @test kernel(A) == ordinary(A, B)
        @test Array(compiled(ra)) == ordinary(A, B)
        @test size(Array(compiled(ra))) == size(ordinary(A, B))
        @test A == [0.2 0.4]
        @test B == [0.3 0.5]
    end
end

@testset "resolved hcat keeps matrix shape and fixed backend structure" begin
    inventories = Any[]
    for n in (17, 33)
        x = collect(range(-0.7, 0.9; length = n))
        u = [0.2, 0.1, 0.3]
        ru = Reactant.to_rarray(u)
        spec = cat_kernel(GlobalRef(Base, :hcat); vector_head = true)
        unbound = prepare(spec)
        kernel = prepare(spec; bound = (; x))
        oracle(v) = sum(abs2, x) * (v[1] * v[2] + v[3])^2 + sum(abs2, v)
        expected = 2u .+ 2sum(abs2, x) * (u[1] * u[2] + u[3]) .* [u[2], u[1], 1.0]
        @test kernel(u) ≈ oracle(u)
        @test first(Enzyme.gradient(Enzyme.Reverse, unbound, u, x)) ≈ expected
        primal = Reactant.@compile kernel(ru)
        reverse(v) = gradient(kernel, v)
        compiled_reverse = Reactant.@compile reverse(ru)
        @test Float64(primal(ru)) ≈ oracle(u)
        @test Array(compiled_reverse(ru)) ≈ expected
        inventory = (operation_names(Reactant.@code_hlo optimize = false kernel(ru)),
            operation_names(Reactant.@code_hlo kernel(ru)),
            operation_names(Reactant.@code_hlo optimize = false reverse(ru)),
            operation_names(Reactant.@code_hlo reverse(ru)))
        @test all(!isempty, inventory)
        push!(inventories, inventory)
    end
    @test inventories[1] == inventories[2]
end

@testset "resolved bindings preserve callable identity" begin
    replacement = ReactiveKernels._tensorized_callee_replacement
    @test replacement(GlobalRef(Custom, :hcat)) === nothing
    @test replacement(:(Custom.hcat), @__MODULE__) === nothing
    @test replacement(GlobalRef(@__MODULE__, :not_defined)) === nothing
    @kernel custom_call(a, x) = begin
        result = Custom.hcat(a, x)
        return result
    end
    kernel = prepare(custom_call; bound = (; x = [0.43]))
    ra = Reactant.to_rarray(0.2; track_numbers = true)
    @test kernel(0.2) ≈ 0.63
    @test Float64((Reactant.@compile kernel(ra))(ra)) ≈ 0.63
end

@traceable make_columns(a, x) = Base.hcat(a, x)
@kernel helper_call(u, x) = begin
    a = sum(u[1:1])
    X = make_columns(a, x)
    loss = sum(abs2, X * u[2:3]) + sum(abs2, u)
    return loss
end

@testset "owned traceable helpers share the resolved lowering" begin
    u, x = [0.2, 0.1, 0.3], [0.43]
    kernel = prepare(helper_call; bound = (; x))
    ru = Reactant.to_rarray(u)
    reverse(v) = gradient(kernel, v)
    @test Float64((Reactant.@compile kernel(ru))(ru)) ≈ kernel(u)
    @test Array((Reactant.@compile reverse(ru))(ru)) ≈ gradient(kernel, u)
end

end # module
