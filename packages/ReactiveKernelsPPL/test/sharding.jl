# Hosted `Run tests` splits this suite across parallel jobs: a single hosted
# job cannot finish it within GitHub's 360-minute job limit (decision
# `ReactiveKernels:hosted-acceptance.02l6qzp/decisions/2026-10-07T12-17-13-677-1i4nqqs`).
#
# `RKPPL_TEST_SHARD=k/N` runs the k-th of N groups of `_PPL_TEST_FILES` with
# about equal measured minutes (`shard_minutes.toml`, `_ppl_shard_assignment`).
# `RKPPL_TEST_FILES=a.jl,b.jl` runs only the named files. Either way, every
# file before the last selected one is still evaluated, with its tests
# removed, because later files reuse its helper functions, constants and
# modules. With neither set, every file runs; setting both is an error.
#
# A failing top-level testset, or a file that throws, is recorded and the run
# continues, so one failure does not hide the results of every later file.
# `_throw_ppl_test_failures` then fails the run and lists every recorded failure.

using TOML

function _ppl_test_shard(spec::AbstractString)
    isempty(spec) && return nothing
    m = match(r"^([0-9]+)/([0-9]+)$", spec)
    m === nothing && error("RKPPL_TEST_SHARD must have the form k/N, got $(repr(spec))")
    k, n = parse(Int, m[1]), parse(Int, m[2])
    1 <= k <= n || error("RKPPL_TEST_SHARD needs 1 <= k <= N, got $(repr(spec))")
    return (k, n)
end

# Measured minutes per test file, from `shard_minutes.toml`.
_ppl_file_minutes(path = joinpath(@__DIR__, "shard_minutes.toml")) =
    Dict{String,Float64}(file => minutes for (file, minutes) in TOML.parsefile(path))

# The shard (1..n) of each file: the longest file first, each to the shard
# with the fewest minutes so far (the lowest such shard on ties). A file
# without a measurement weighs the median measurement.
function _ppl_shard_assignment(files, n::Int, minutes::AbstractDict)
    known = sort!(collect(values(minutes)))
    unlisted = isempty(known) ? 1.0 : known[cld(length(known), 2)]
    weight(i) = get(minutes, files[i], unlisted)
    load, shard = zeros(n), zeros(Int, length(files))
    for i in sort!(collect(eachindex(files)); by = i -> (-weight(i), i))
        shard[i] = argmin(load)
        load[shard[i]] += weight(i)
    end
    return shard
end

function _ppl_test_names(spec::AbstractString)
    isempty(spec) && return nothing
    names = String.(strip.(split(spec, ',')))
    any(isempty, names) && error("RKPPL_TEST_FILES must be comma-separated file names, got $(repr(spec))")
    return names
end

# The files to evaluate and the indices of those that run their tests, for
# the RKPPL_TEST_* settings in `env`.
function _ppl_test_plan(files, env; minutes = _ppl_file_minutes())
    shard = _ppl_test_shard(get(env, "RKPPL_TEST_SHARD", ""))
    names = _ppl_test_names(get(env, "RKPPL_TEST_FILES", ""))
    shard === nothing || names === nothing ||
        error("set RKPPL_TEST_SHARD or RKPPL_TEST_FILES, not both")
    kept = collect(String, files)
    if names === nothing
        shard === nothing && return kept, collect(eachindex(kept))
        k, n = shard
        return kept, findall(==(k), _ppl_shard_assignment(kept, n, minutes))
    end
    selected = map(names) do name
        i = findfirst(==(name), kept)
        i === nothing || return i
        error("$name is not in _PPL_TEST_FILES")
    end
    return kept, sort!(unique!(selected))
end

_macro_name(name::Symbol) = name
_macro_name(name::GlobalRef) = name.name
_macro_name(name::QuoteNode) = _macro_name(name.value)
_macro_name(name::Expr) = Meta.isexpr(name, :.) ? _macro_name(name.args[end]) : Symbol()
_macro_name(_) = Symbol()

_is_test_macrocall(ex) =
    Meta.isexpr(ex, :macrocall) && startswith(String(_macro_name(ex.args[1])), "@test")

# Applies `f` to every top-level `@test…` call, inside module bodies and
# nested includes too (which `include` through `mapexpr`), and keeps every
# other top-level expression.
function _map_tests(mapexpr, f, ex)
    _is_test_macrocall(ex) && return f(ex)
    if Meta.isexpr(ex, :module)
        return Expr(:module, ex.args[1], ex.args[2], _map_tests(mapexpr, f, ex.args[3]))
    elseif Meta.isexpr(ex, (:block, :toplevel, :||, :&&))
        return Expr(ex.head, (_map_tests(mapexpr, f, a) for a in ex.args)...)
    elseif Meta.isexpr(ex, :call) && ex.args[1] === :include && length(ex.args) == 2
        return Expr(:call, :include, mapexpr, ex.args[2])
    end
    return ex
end

# `include` map: drops top-level `@test…` calls.
_without_tests(ex) = _map_tests(_without_tests, _ -> nothing, ex)

# `include` map: runs each top-level `@test…` call, recording a thrown failure
# as `label => exception` in `failures` instead of stopping the include.
function _continuing_tests(failures, label)
    record(test) = :(try
        $test
    catch err
        $push!($failures, $Pair($label, err))
        nothing
    end)
    mapexpr(ex) = _map_tests(mapexpr, record, ex)
    return mapexpr
end

# Returns the recorded `label => exception` failures.
function _run_ppl_test_files(all_files, env)
    files, selected = _ppl_test_plan(all_files, env)
    length(selected) < length(files) &&
        println("RKPPL: running $(length(selected)) of $(length(files)) test files")
    failures = Pair{String,Any}[]
    for i in 1:(isempty(selected) ? 0 : last(selected))
        path = joinpath(@__DIR__, files[i])
        label = "test file $i/$(length(files)): $(files[i])"
        try
            if i in selected
                println("RKPPL ", label)
                started = time()
                include(_continuing_tests(failures, label), path)
                println("RKPPL ", label, " took ", round(time() - started; digits = 1), " s")
            else
                include(_without_tests, path)
            end
        catch err
            # The rest of this file did not run; later files may miss its definitions.
            println("RKPPL $label stopped: ", sprint(showerror, err))
            push!(failures, label => err)
        end
    end
    return failures
end

function _throw_ppl_test_failures(failures)
    isempty(failures) && return nothing
    for (label, err) in failures
        println("RKPPL failure in ", label, ": ", sprint(showerror, err))
    end
    error("$(length(failures)) RKPPL test failure(s) in ",
        join(unique(first.(failures)), "; "))
end
