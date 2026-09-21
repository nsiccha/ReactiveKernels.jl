# Syntax-drift corpus guard: committed canonical surface programs pin the
# surface→plan lowering. Each test/corpus/NN_name.jl holds one canonical
# program (first line `# data: ...` names the data columns);
# test/corpus/golden/NN_name.canon holds the canonical serialization of its
# unbound plan. Any lowering drift fails loudly; re-bless deliberately with
# RKPPL_REBLESS=1 after review, never to make red green.

const _CORPUS_DIR = joinpath(@__DIR__, "corpus")
const _CORPUS_GOLDEN_DIR = joinpath(_CORPUS_DIR, "golden")

# Generic canonical serializer: fixed field order, sorted dict keys, no
# line numbers, no memory addresses. New IR fields change the output, which
# is exactly what this guard must catch.
function _canon(io::IO, x, depth::Int = 0)
    depth > 60 && (print(io, "<depth>"); return)
    if x === nothing || x === missing
        print(io, repr(x))
    elseif x isa Union{Bool, Symbol, Number, Char, String}
        print(io, repr(x))
    elseif x isa LineNumberNode
        print(io, "<ln>") # dropped by the Expr branch; unreachable otherwise
    elseif x isa Expr
        print(io, "Expr(:", x.head)
        for a in x.args
            a isa LineNumberNode && continue
            print(io, " ")
            _canon(io, a, depth + 1)
        end
        print(io, ")")
    elseif x isa QuoteNode
        print(io, "quote(")
        _canon(io, x.value, depth + 1)
        print(io, ")")
    elseif x isa AbstractUnitRange
        print(io, repr(first(x)), ":", repr(last(x)))
    elseif x isa NamedTuple
        print(io, "(")
        for (i, k) in enumerate(keys(x))
            i > 1 && print(io, " ")
            print(io, k, "=")
            _canon(io, x[k], depth + 1)
        end
        print(io, ")")
    elseif x isa Tuple
        print(io, "(")
        for (i, v) in enumerate(x)
            i > 1 && print(io, " ")
            _canon(io, v, depth + 1)
        end
        print(io, ")")
    elseif x isa AbstractArray
        print(io, "[")
        for (i, v) in enumerate(x)
            i > 1 && print(io, " ")
            _canon(io, v, depth + 1)
        end
        print(io, "]")
    elseif x isa AbstractDict
        print(io, "{")
        ks = sort!(collect(keys(x)); by = repr)
        for (i, k) in enumerate(ks)
            i > 1 && print(io, " ")
            _canon(io, k, depth + 1)
            print(io, "=>")
            _canon(io, x[k], depth + 1)
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
        if isempty(fs)
            print(io, repr(x))
        else
            print(io, nameof(typeof(x)), "(")
            for (i, f) in enumerate(fs)
                i > 1 && print(io, " ")
                print(io, f, "=")
                _canon(io, getfield(x, f), depth + 1)
            end
            print(io, ")")
        end
    end
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

@testset "corpus drift guard" begin
    cases = sort!(filter(f -> endswith(f, ".jl"),
        readdir(_CORPUS_DIR; join = true)))
    @test !isempty(cases)
    rebless = get(ENV, "RKPPL_REBLESS", "") == "1"
    for path in cases
        name = splitext(basename(path))[1]
        @testset "$name" begin
            ast, data = _load_corpus_case(path)
            plan = lower_rkppl(ast, data)
            canon = sprint(_canon, plan) * "\n"
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
        inline_plan = lower_rkppl(inline, data)
        bound_plan = lower_rkppl(bound, data)
        @test sprint(_canon, bound_plan) == sprint(_canon, inline_plan)
        @test length(bound_plan.levelmaps) == 2
        @test all(m -> m.subset == subset, bound_plan.levelmaps)
    end
end
