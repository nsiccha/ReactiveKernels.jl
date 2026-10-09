# Syntax-drift corpus guard: committed canonical surface programs pin the
# surface→plan lowering. Each test/corpus/NN_name.jl holds one canonical
# program (first line `# data: ...` names the data columns);
# test/corpus/golden/NN_name.canon holds the canonical serialization of its
# unbound plan. Any lowering drift fails loudly; re-bless deliberately with
# RKPPL_REBLESS=1 after review, never to make red green.

import ReactiveKernelsDistributionKernels

const _CORPUS_DIR = joinpath(@__DIR__, "corpus")
const _CORPUS_GOLDEN_DIR = joinpath(_CORPUS_DIR, "golden")

# Generic canonical serializer: fixed field order, sorted dict keys, no
# line numbers, no memory addresses. New meaningful IR fields change the
# output; empty optional metadata preserves existing snapshots.
function _canon(io::IO, x, depth::Int = 0, names = nothing)
    depth > 60 && (print(io, "<depth>"); return)
    if x === nothing || x === missing
        print(io, repr(x))
    elseif x isa Symbol && names !== nothing && _corpus_private_selected(x, names) !== nothing
        prefix, binder = _corpus_private_selected(x, names)
        index = get!(names.selected, binder, length(names.selected) + 1)
        print(io, isempty(prefix) ? "private_selected(" : "private_selected_index(", index, ")")
    elseif x isa Union{Bool, Symbol, Number, Char, String}
        print(io, repr(x))
    elseif x isa LineNumberNode
        print(io, "<ln>") # dropped by the Expr branch; unreachable otherwise
    elseif x isa Expr
        print(io, "Expr(:", x.head)
        for a in x.args
            a isa LineNumberNode && continue
            print(io, " ")
            _canon(io, a, depth + 1, names)
        end
        print(io, ")")
    elseif x isa GlobalRef
        # A resolved module function (functions as values): module + name
        # only (the runtime binding object is not part of the plan).
        print(io, "GlobalRef(", nameof(x.mod), ".", x.name, ")")
    elseif x isa QuoteNode
        print(io, "quote(")
        _canon(io, x.value, depth + 1, names)
        print(io, ")")
    elseif x isa AbstractUnitRange
        print(io, repr(first(x)), ":", repr(last(x)))
    elseif x isa NamedTuple
        print(io, "(")
        for (i, k) in enumerate(keys(x))
            i > 1 && print(io, " ")
            print(io, k, "=")
            _canon(io, x[k], depth + 1, names)
        end
        print(io, ")")
    elseif x isa Tuple
        print(io, "(")
        for (i, v) in enumerate(x)
            i > 1 && print(io, " ")
            _canon(io, v, depth + 1, names)
        end
        print(io, ")")
    elseif x isa AbstractArray
        print(io, "[")
        for (i, v) in enumerate(x)
            i > 1 && print(io, " ")
            _canon(io, v, depth + 1, names)
        end
        print(io, "]")
    elseif x isa AbstractDict
        print(io, "{")
        ks = sort!(collect(keys(x)); by = repr)
        for (i, k) in enumerate(ks)
            i > 1 && print(io, " ")
            _canon(io, k, depth + 1, names)
            print(io, "=>")
            _canon(io, x[k], depth + 1, names)
        end
        print(io, "}")
    elseif x isa Type
        print(io, nameof(x))
    elseif x isa Function
        print(io, nameof(x))
    elseif x isa Module
        print(io, nameof(x))
    else
        fs = try
            fieldnames(typeof(x))
        catch
            ()
        end
        # Empty lexical metadata changes no existing corpus model. Nonempty
        # scopes remain serialized and require their own reviewed goldens.
        x isa StructuralPlan && isempty(x.submodel_scopes) &&
            (fs = filter(!=(:submodel_scopes), fs))
        x isa LikelihoodSpec && x.threshold_effects === nothing &&
            (fs = filter(!=(:threshold_effects), fs))
        x isa LikelihoodSpec && isempty(x.mixture_trials) &&
            (fs = filter(!=(:mixture_trials), fs))
        x isa StructuralPlan && isempty(x.conditioned) &&
            (fs = filter(!=(:conditioned), fs))
        x isa StructuralPlan && isempty(x.external_observations) &&
            (fs = filter(!=(:external_observations), fs))
        x isa StructuralPlan && isempty(x.cell_broadcasts) &&
            (fs = filter(!=(:cell_broadcasts), fs))
        x isa VectorParameter && x.extent_expr === nothing &&
            (fs = filter(!=(:extent_expr), fs))
        # Canonical mathematical plans compare loop/broadcast spellings.
        # Their indexing admission metadata is tested independently.
        x isa StructuralPlan &&
            (fs = filter(!=(:indexed_observations), fs))
        # The authored names of optimized definitions are queryability
        # metadata, pinned by `test_authored_names.jl`.
        x isa StructuralPlan &&
            (fs = filter(!=(:named_values), fs))

        if isempty(fs)
            print(io, repr(x))
        else
            print(io, nameof(typeof(x)), "(")
            for (i, f) in enumerate(fs)
                i > 1 && print(io, " ")
                print(io, f, "=")
                _canon(io, getfield(x, f), depth + 1, names)
            end
            print(io, ")")
        end
    end
end

# Only private selected-plate binders are alpha-equivalent. Authored symbols,
# including quoted names with the same spelling, remain literal; numeric types
# and every other plan field keep the generic serializer's representation.
function _corpus_authored_names!(names, x)
    if x isa Symbol
        push!(names, x)
    elseif x isa Expr
        foreach(a -> _corpus_authored_names!(names, a), x.args)
    elseif x isa QuoteNode
        _corpus_authored_names!(names, x.value)
    end
    return names
end

function _corpus_private_selected(x::Symbol, names)
    x in names.authored && return nothing
    match_name = match(r"^(_ppl_pi_)?(##_rkppl_selected#\d+)$", String(x))
    match_name === nothing && return nothing
    binder = Symbol(match_name.captures[2])
    binder in names.authored && return nothing
    return something(match_name.captures[1], ""), binder
end

function _corpus_canon(io::IO, plan, ast)
    names = (; authored = _corpus_authored_names!(Set{Symbol}(), ast),
        selected = Dict{Symbol,Int}())
    _canon(io, plan, 0, names)
end

function _load_corpus_case(path::String)
    lines = readlines(path)
    isempty(lines) && error("empty corpus file: $path")
    m = match(r"^#\s*data:\s*(.*?)\s*$", lines[1])
    m === nothing && error("corpus $path: first line must be `# data: ...`")
    data = Tuple(Symbol(s) for s in split(m.captures[1]))
    ast = Meta.parse(join(lines[2:end], "\n"))
    ast isa Expr && ast.head === :block ||
        error("corpus $path: body must parse to a `begin ... end` block")
    return ast, data
end

function _first_diff(a::String, b::String)
    n = min(length(a), length(b))
    for i in 1:n
        a[i] != b[i] && return i
    end
    return n + 1
end

@testset "optional response metadata serialization" begin
    plan = lower_rkppl(quote
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:y,); conditioned = (:y,))
    response = only(plan.responses)
    @test !occursin("threshold_effects=", sprint(_canon, response))
    effects = ReactiveKernelsPPL._with(response; threshold_effects = :effects)
    other = ReactiveKernelsPPL._with(response; threshold_effects = :other_effects)
    @test occursin("threshold_effects=:effects", sprint(_canon, effects))
    @test sprint(_canon, effects) != sprint(_canon, other)
    @test !occursin("mixture_trials=", sprint(_canon, response))
    trials = ReactiveKernelsPPL._with(response; mixture_trials = Any[2, 3])
    other_trials = ReactiveKernelsPPL._with(response; mixture_trials = Any[2, 4])
    @test occursin("mixture_trials=[2 3]", sprint(_canon, trials))
    @test sprint(_canon, trials) != sprint(_canon, other_trials)
end

@testset "optional plan metadata serialization" begin
    cell(rhs) = quote
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            $rhs
        end
    end
    plain = lower_rkppl(cell(:(y[i] ~ Normal(x[i], sigma))), (:y, :x); conditioned = (:y,))
    nested = lower_rkppl(cell(:(y[i] .~ Normal.(x[i], sigma))), (:y, :x); conditioned = (:y,))
    @test isempty(plain.cell_broadcasts)
    @test !occursin("cell_broadcasts=", sprint(_canon, plain))
    @test !isempty(nested.cell_broadcasts)
    @test occursin("cell_broadcasts={:y=>[:x]}", sprint(_canon, nested))
end

@testset "corpus private binder alpha equivalence" begin
    authored = Symbol("##_rkppl_selected#17")
    ast = Expr(:block, QuoteNode(authored))
    first_names = Expr(:tuple, Symbol("##_rkppl_selected#21"), authored,
        Symbol("##_rkppl_selected#21"), Symbol("##_rkppl_selected#22"))
    later_names = Expr(:tuple, Symbol("##_rkppl_selected#101"), authored,
        Symbol("##_rkppl_selected#101"), Symbol("##_rkppl_selected#102"))
    canonical = sprint(_corpus_canon, first_names, ast)
    @test canonical == sprint(_corpus_canon, later_names, ast)
    @test occursin(repr(authored), canonical)
    @test count("private_selected(1)", canonical) == 2
    @test count("private_selected(2)", canonical) == 1
    indexed = Expr(:tuple, Symbol("##_rkppl_selected#21"),
        Symbol("_ppl_pi_##_rkppl_selected#21"))
    @test sprint(_corpus_canon, indexed, ast) ==
        "Expr(:tuple private_selected(1) private_selected_index(1))"
    @test sprint(_corpus_canon, 1, ast) != sprint(_corpus_canon, 1.0, ast)
    @test sprint(_corpus_canon, Symbol("user21"), ast) !=
        sprint(_corpus_canon, Symbol("user101"), ast)

    selected_ast, data = _load_corpus_case(joinpath(_CORPUS_DIR,
        "99_plate_73_beta.jl"))
    first_plan = lower_rkppl(selected_ast, data; conditioned = data)
    for _ in 1:7
        gensym(:_rkppl_selected)
    end
    later_plan = lower_rkppl(selected_ast, data; conditioned = data)
    @test sprint(_canon, first_plan) != sprint(_canon, later_plan)
    @test sprint(_corpus_canon, first_plan, selected_ast) ==
        sprint(_corpus_canon, later_plan, selected_ast)
end

@testset "corpus drift guard" begin
    cases = sort!(filter(f -> endswith(f, ".jl"),
        readdir(_CORPUS_DIR; join = true)))
    @test !isempty(cases)
    rebless = get(ENV, "RKPPL_REBLESS", "") == "1"
    for path in cases
        name = splitext(basename(path))[1]
        @testset "$name" begin
            ast, data = _load_corpus_case(path)
            plan = lower_rkppl(ast, data; conditioned = data)
            canon = sprint(_corpus_canon, plan, ast) * "\n"
            golden = joinpath(_CORPUS_GOLDEN_DIR, name * ".canon")
            if rebless
                mkpath(_CORPUS_GOLDEN_DIR)
                write(golden, canon)
                @test true
            else
                @test isfile(golden)
                if isfile(golden)
                    want = read(golden, String)
                    @test canon == want
                    if canon != want
                        i = _first_diff(canon, want)
                        @info "corpus drift" case = name first_diff_at = i got_snippet = canon[max(1, i - 80):min(length(canon), i + 80)] want_snippet = want[max(1, i - 80):min(length(want), i + 80)]
                    end
                end
            end
        end
    end
end

@testset "bound levels spelling canonical parity" begin
    # Every subset shape admitted inline is admitted through a bound name,
    # and the unbound plan is byte-identical under the canonical serializer.
    two_to_end = Expr(:call, :(:), 2, :end)
    cases = (
        (:(levels(g)), :(levels(h)), nothing, Colon()),
        (Expr(:ref, :(levels(g)), two_to_end),
            Expr(:ref, :(levels(h)), two_to_end), two_to_end, (2, :end)),
        (:(levels(g)[1:2]), :(levels(h)[1:2]), :(1:2), 1:2),
        (:(levels(g)[[1, 3]]), :(levels(h)[[1, 3]]), :([1, 3]), [1, 3]),
    )
    for (g_rhs, h_rhs, subset_expr, subset) in cases
        g_index = g_rhs.head === :call && subset_expr !== nothing ?
            Expr(:ref, g_rhs, subset_expr) : g_rhs
        h_index = h_rhs.head === :call && subset_expr !== nothing ?
            Expr(:ref, h_rhs, subset_expr) : h_rhs
        inline = Expr(:block,
            Expr(:call, :.~, Expr(:ref, :c, g_index), :(Normal.(0, 2))),
            Expr(:call, :.~, Expr(:ref, :k, h_index), :(Normal.(0, 3))),
            :(mu = c[g] .+ k[h]),
            Expr(:call, :.~, :y, :(Normal.(mu, 1.5))))
        bound = Expr(:block, Expr(:(=), :sel_g, g_rhs),
            Expr(:(=), :sel_h, h_rhs),
            Expr(:call, :.~, Expr(:ref, :c, :sel_g), :(Normal.(0, 2))),
            Expr(:call, :.~, Expr(:ref, :k, :sel_h), :(Normal.(0, 3))),
            :(mu = c[g] .+ k[h]),
            Expr(:call, :.~, :y, :(Normal.(mu, 1.5))))
        data = (:y, :g, :h)
        inline_plan = lower_rkppl(inline, data; conditioned = data)
        bound_plan = lower_rkppl(bound, data; conditioned = data)
        @test sprint(_canon, bound_plan) == sprint(_canon, inline_plan)
        @test length(bound_plan.levelmaps) == 2
        @test all(m -> m.subset == subset, bound_plan.levelmaps)
    end
end
