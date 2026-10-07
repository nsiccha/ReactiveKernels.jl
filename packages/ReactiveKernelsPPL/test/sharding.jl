# Hosted `Run tests` splits this suite across parallel jobs: a single hosted
# job cannot finish it within GitHub's 360-minute job limit (decision
# `ReactiveKernels:hosted-acceptance.02l6qzp/decisions/2026-10-07T12-17-13-677-1i4nqqs`).
#
# `RKPPL_TEST_SHARD=k/N` runs files k, k + N, k + 2N, … of `_PPL_TEST_FILES`.
# Every file before the last selected one is still evaluated, with its tests
# removed, because later files reuse its helper functions, constants and
# modules. An unset or empty value runs every file.

function _ppl_test_shard(spec::AbstractString)
    isempty(spec) && return nothing
    m = match(r"^([0-9]+)/([0-9]+)$", spec)
    m === nothing && error("RKPPL_TEST_SHARD must have the form k/N, got $(repr(spec))")
    k, n = parse(Int, m[1]), parse(Int, m[2])
    1 <= k <= n || error("RKPPL_TEST_SHARD needs 1 <= k <= N, got $(repr(spec))")
    return (k, n)
end

_ppl_shard_selects(::Int, ::Nothing) = true
_ppl_shard_selects(i::Int, (k, n)::Tuple{Int,Int}) = mod1(i, n) == k

_macro_name(name::Symbol) = name
_macro_name(name::GlobalRef) = name.name
_macro_name(name::QuoteNode) = _macro_name(name.value)
_macro_name(name::Expr) = Meta.isexpr(name, :.) ? _macro_name(name.args[end]) : Symbol()
_macro_name(_) = Symbol()

_is_test_macrocall(ex) =
    Meta.isexpr(ex, :macrocall) && startswith(String(_macro_name(ex.args[1])), "@test")

# `include` map: drops top-level `@test…` calls, inside module bodies and
# nested includes too, and keeps every other top-level expression.
function _without_tests(ex)
    _is_test_macrocall(ex) && return nothing
    if Meta.isexpr(ex, :module)
        return Expr(:module, ex.args[1], ex.args[2], _without_tests(ex.args[3]))
    elseif Meta.isexpr(ex, (:block, :toplevel, :||, :&&))
        return Expr(ex.head, map(_without_tests, ex.args)...)
    elseif Meta.isexpr(ex, :call) && ex.args[1] === :include && length(ex.args) == 2
        return Expr(:call, :include, _without_tests, ex.args[2])
    end
    return ex
end

function _run_ppl_test_files(files, spec::AbstractString)
    shard = _ppl_test_shard(spec)
    selected = findall(i -> _ppl_shard_selects(i, shard), eachindex(files))
    shard === nothing ||
        println("RKPPL_TEST_SHARD=$spec: running $(length(selected)) of $(length(files)) test files")
    for i in 1:(isempty(selected) ? 0 : last(selected))
        path = joinpath(@__DIR__, files[i])
        if i in selected
            println("RKPPL test file $i/$(length(files)): $(files[i])")
            include(path)
        else
            include(_without_tests, path)
        end
    end
end
