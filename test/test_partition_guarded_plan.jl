using ReactiveKernels, Enzyme, Test
const _PGP = ReactiveKernels

# A compiled program's kept expression is `(args...) -> body`, and a generator
# left in the body is already `Base.Generator(@opaque (…) -> …, …)`: a closure
# shows as a lambda there.
_pgp_has_closure(x) = x isa Expr && x.head !== :quote &&
    (x.head in (:generator, :->) || any(_pgp_has_closure, x.args))
_pgp_concrete_storage(x) = isconcretetype(typeof(x))
_pgp_concrete_storage(x::Tuple) = all(_pgp_concrete_storage, x)
_pgp_concrete_storage(x::AbstractArray) =
    isconcretetype(eltype(x)) && all(_pgp_concrete_storage, x)

function _pgp_check(body, cases; active_values = (2.0, 3.0, 4.0), require_plan = true)
    split = _PGP._partition_plan_split(body, [:u], Any[])
    require_plan && @test split !== nothing
    # A conservative split may leave the complete body in ordinary execution.
    # Test its semantics too, without requiring an optimization for every form.
    plan, main, needed, _ = split === nothing ?
        (Any[], body.args, Symbol[], Symbol[]) : split
    planner = _PGP.compile(Expr(:function, Expr(:tuple, :u, :label),
        Expr(:block, plan..., Expr(:return, Expr(:tuple, needed...)))))
    # `_compile` lifts the cache's comprehension out of the compiled body: an
    # opaque closure there cannot run while a package image is generated
    # (`test_precompiled_guarded_plan.jl`).
    @test !_pgp_has_closure(
        _PGP._program_expression(_PGP._native_generated_function(planner)).args[2])
    reads = [Expr(:(=), name, :(p[$i])) for (i, name) in enumerate(needed)]
    planned = _PGP.compile(Expr(:function, Expr(:tuple, :u, :label, :p),
        Expr(:block, reads..., main...)))
    plain = _PGP.compile(Expr(:function, Expr(:tuple, :u, :label), body))
    types = Type[]
    for label in cases
        p = planner(-2.0, label)
        @test _pgp_concrete_storage(p)
        push!(types, typeof(p))
        before = deepcopy(p)
        for u in active_values
            @test planned(u, label, p) == plain(u, label)
            @test isequal(p, before)
            got = first(first(Enzyme.autodiff(Enzyme.Reverse, planned, Enzyme.Active,
                Enzyme.Active(u), Enzyme.Const(label), Enzyme.Const(p))))
            expected = first(first(Enzyme.autodiff(Enzyme.Reverse, plain, Enzyme.Active,
                Enzyme.Active(u), Enzyme.Const(label))))
            @test got ≈ expected
            @test isequal(p, before)
        end
    end
    @test all(==(first(types)), types)
    @test isconcretetype(first(types))
    (; plan, main, needed, planner, planned, plain)
end

@testset "guarded plan lexical bindings" begin
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = let r = rows
                r[[1]]
            end
            y = u * sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
    # A let RHS reads the outer binding even when its new local has that name.
    # The existing alias analysis may retain this whole body; either execution
    # path must keep those lexical identities and ordinary derivatives.
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            firstrow = rows[[1]]
            selected = let firstrow = firstrow, rows = rows
                firstrow .+ rows[[1]]
            end
            y = u * sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31); require_plan = false)
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = let r = rows, s = r[[1]]
                s
            end
            y = let x = u
                log(x) * sum(selected)
            end
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = let u = rows
                u[[1]]
            end
            y = u * sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = let r = rows
                r[[1]]
            end
            scratch = zeros(2)
            alias = let s = scratch
                view(s, :)
            end
            alias[1] = u
            y = sum(alias) + sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
end

@testset "guarded plans retain runtime iteration and guard semantics" begin
    # The selected row construction must leave the value body. This is the
    # source-call shape responsible for repeated integer-vector allocations.
    split = _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = let r = rows
                ReactiveKernels._native_vcat(r)
            end
            y = u * sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
    has_head(x, head) = x isa Expr &&
        (x.head === head || any(a -> has_head(a, head), x.args))
    has_call(x, name) = x isa Expr &&
        ((x.head === :call && _PGP._plan_call_name(x.args[1]) === name) ||
         any(a -> has_call(a, name), x.args))
    @test has_call(Expr(:block, split.plan...), :_native_vcat)
    @test !has_call(Expr(:block, split.main...), :_native_vcat)
    @test has_head(Expr(:block, split.plan...), :generator)
    @test has_head(Expr(:block, split.main...), :if)
    # One compiled planner/body serves empty and increasing runtime lengths;
    # no expression or trace is generated for an individual length here.

    # An active guard must change with each call, not with the planning exemplar.
    _pgp_check(quote
        rows = collect(1:label)
        if u > 0 && !isempty(rows)
            selected = rows[[1]]
            y = u * sum(selected)
        else
            y = -u
        end
        return y
    end, (0, 1, 4, 31); active_values = (-3.0, 2.0, 4.0))

    # Preserve Julia's Bool condition check instead of converting a condition
    # to the cached comprehension's integer trip count.
    guarded = _pgp_check(quote
        rows = collect(1:label)
        if label
            selected = rows[[1]]
            y = u * sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (false, true))
    @test_throws TypeError guarded.plain(2.0, 1)
    @test_throws TypeError guarded.planner(-2.0, 1)
end
@testset "guarded plans preserve branches, active values and ownership" begin
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = rows[[1]]
            y = u * sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = rows[[1]]
            y = log(u) * sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = rows[[1]]
            scratch = zeros(2)
            alias = view(scratch, :)
            alias[1] = u
            y = sum(alias) + sum(selected)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
    _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            selected = rows[[1]]
            if label > 1
                indices = collect(1:label)
                y = u * sum(selected) * sum(indices)
            else
                y = u * sum(selected)
            end
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 4, 31))
end

@testset "guarded plans hold rows of an active value's shape that a later guard reads" begin
    # An arm computes a value from `u` and rows from its shape alone, as the
    # guard several partition members share computes rows from a product's
    # shape, whose factor is computed from the parameters. The plan holds the
    # shape and those rows; the arm checks the shape against the value's on
    # every call. A later guard reads the rows, so the arm binds them from the
    # plan too (snag test-plate-parti-13fa3aa6).
    split = _pgp_check(quote
        rows = collect(1:label)
        if !isempty(rows)
            w = [u, 2u, 3u]
            shape = axes(w)
            positions = getindex(collect(first(shape)), rows)
            v = sum(getindex(w, positions))
        end
        if !isempty(rows)
            y = v * length(positions)
        else
            y = zero(u)
        end
        return y
    end, (0, 1, 2, 3))
    main = Expr(:block, split.main...)
    has_call(x, name) = x isa Expr &&
        ((x.head === :call && _PGP._plan_call_name(x.args[1]) === name) ||
         any(a -> has_call(a, name), x.args))
    @test has_call(main, :_cell_plan_check)
    @test !has_call(main, :collect)
    @test has_call(Expr(:block, split.plan...), :collect)
end

module _PGPScanAllocatorLookalike
_scan_similar_output(rows, ::Type{T}) where {T} = rows
end

@testset "fresh scan buffers retain marker planning" begin
    split = _pgp_check(quote
        rows = collect(1:label)
        concrete = isconcretetype(Float64)
        preallocated = concrete ? $(GlobalRef(ReactiveKernels, :_scan_similar_output))(rows, Float64) : nothing
        scratch = preallocated === nothing ? $(GlobalRef(ReactiveKernels, :_scan_similar_output))(rows, Float64) : preallocated
        for i in eachindex(scratch)
            scratch[i] = u * rows[i]
        end
        return sum(scratch)
    end, (0, 1, 4, 31))
    @test :rows in split.needed
    @test !(:scratch in split.needed)
    @test !(:preallocated in split.needed)
end

@testset "conditional aliases retain mutable backing arrays" begin
    _pgp_check(quote
        a = zeros(label + 1)
        b = zeros(label + 1)
        selected = isodd(label) ? a : b
        selected[1] = u
        return sum(a) + 2sum(b)
    end, (0, 1, 4, 31); require_plan=false)

    # The allocator promise belongs to an exact owned callable. An unrelated
    # function with the same name may return its marker as an alias.
    _pgp_check(quote
        rows = zeros(label + 1)
        scratch = $(GlobalRef(_PGPScanAllocatorLookalike, :_scan_similar_output))(rows, Float64)
        scratch[1] = u
        return sum(rows)
    end, (0, 1, 4, 31); require_plan=false)

    # A condition does not supply the returned storage, but evaluating it may
    # still mutate an array. Such writes must remain in ordinary execution.
    _pgp_check(quote
        rows = zeros(label + 1)
        scratch = (rows[1] = u; isodd(label)) ? rows : rows
        return sum(scratch)
    end, (0, 1, 4, 31); require_plan=false)
end
