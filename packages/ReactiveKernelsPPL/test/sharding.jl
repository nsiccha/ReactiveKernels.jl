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
# `RKPPL_TEST_BACKENDS=native` leaves out every Reactant file (a file whose
# name contains `reactant`), so the suite runs in an environment without
# Reactant, e.g. (`test/runtests.jl` with an environment that develops this
# package and adds its dependencies and test dependencies except Reactant):
#
#     RKPPL_TEST_BACKENDS=native RKPPL_TEST_FILES=test_mi.jl julia --project=<env> test/runtests.jl
#
# Shards then partition the native files. Native files never use Reactant
# or a definition from a Reactant file; `_ppl_native_reactant_uses` checks
# that. `RKPPL_TEST_BACKENDS=all`, or unset, keeps every file.
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
# without a measurement weighs the median of its kind (Reactant or native).
function _ppl_shard_assignment(files, n::Int, minutes::AbstractDict)
    function median_minutes(reactant)
        known = sort!([m for (file, m) in minutes if _ppl_reactant_file(file) == reactant])
        return isempty(known) ? 1.0 : known[cld(length(known), 2)]
    end
    unlisted = Dict(kind => median_minutes(kind) for kind in (false, true))
    weight(i) = get(minutes, files[i], unlisted[_ppl_reactant_file(files[i])])
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

function _ppl_test_backends(spec::AbstractString)
    spec in ("", "all") && return :all
    spec == "native" && return :native
    error("RKPPL_TEST_BACKENDS must be `all` or `native`, got $(repr(spec))")
end

_ppl_reactant_file(name::AbstractString) = occursin("reactant", name)

# The files to evaluate and the indices of those that run their tests, for
# the RKPPL_TEST_* settings in `env`.
function _ppl_test_plan(files, env; minutes = _ppl_file_minutes())
    backends = _ppl_test_backends(get(env, "RKPPL_TEST_BACKENDS", ""))
    shard = _ppl_test_shard(get(env, "RKPPL_TEST_SHARD", ""))
    names = _ppl_test_names(get(env, "RKPPL_TEST_FILES", ""))
    shard === nothing || names === nothing ||
        error("set RKPPL_TEST_SHARD or RKPPL_TEST_FILES, not both")
    kept = String[f for f in files if backends === :all || !_ppl_reactant_file(f)]
    if names === nothing
        shard === nothing && return kept, collect(eachindex(kept))
        k, n = shard
        return kept, findall(==(k), _ppl_shard_assignment(kept, n, minutes))
    end
    selected = map(names) do name
        i = findfirst(==(name), kept)
        i === nothing || return i
        name in files && error("$name is a Reactant file, which RKPPL_TEST_BACKENDS=native leaves out")
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
    length(files) < length(all_files) &&
        println("RKPPL_TEST_BACKENDS=native: leaving out $(length(all_files) - length(files)) Reactant test files")
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

# Names that a top-level expression defines.
_ppl_defined_names(ex) = _ppl_defined_names!(Set{Symbol}(), ex)
function _ppl_defined_names!(names, ex)
    ex isa Expr || return names
    if Meta.isexpr(ex, (:function, :macro, :(=)))
        target = ex.args[1]
        while Meta.isexpr(target, (:where, :(::), :call, :curly))
            target = target.args[1]
        end
        target isa Symbol && push!(names, target)
        Meta.isexpr(target, :tuple) && union!(names, filter(a -> a isa Symbol, target.args))
    elseif Meta.isexpr(ex, (:const, :global, :block, :toplevel)) ||
            Meta.isexpr(ex, :macrocall) && !_is_test_macrocall(ex)
        foreach(a -> _ppl_defined_names!(names, a), ex.args)
    elseif Meta.isexpr(ex, :module)
        push!(names, ex.args[2])
    elseif Meta.isexpr(ex, :struct)
        target = ex.args[2]
        while Meta.isexpr(target, (:<:, :curly))
            target = target.args[1]
        end
        push!(names, target)
    end
    return names
end

_ppl_symbols!(symbols, ex::Symbol) = push!(symbols, ex)
_ppl_symbols!(symbols, ex::QuoteNode) = _ppl_symbols!(symbols, ex.value)
_ppl_symbols!(symbols, ex::Expr) = (foreach(a -> _ppl_symbols!(symbols, a), ex.args); symbols)
_ppl_symbols!(symbols, _) = symbols

# The literal `include("…")` / `include(joinpath(@__DIR__, "…", …))` paths of
# a top-level expression, relative to `dir`.
function _ppl_included_paths(ex, dir)
    paths = String[]
    ex isa Expr || return paths
    if Meta.isexpr(ex, :call) && ex.args[1] === :include && length(ex.args) == 2
        arg = ex.args[2]
        arg isa String && push!(paths, joinpath(dir, arg))
        if Meta.isexpr(arg, :call) && arg.args[1] === :joinpath &&
                Meta.isexpr(arg.args[2], :macrocall) && all(a -> a isa String, arg.args[3:end])
            push!(paths, joinpath(dir, arg.args[3:end]...))
        end
    end
    foreach(a -> append!(paths, _ppl_included_paths(a, dir)), ex.args)
    return paths
end

# Top-level expressions of `path` and of the files it includes literally.
function _ppl_file_expressions(path)
    exs = filter(ex -> !(ex isa LineNumberNode), Meta.parseall(read(path, String); filename=path).args)
    for ex in copy(exs), included in _ppl_included_paths(ex, dirname(path))
        append!(exs, _ppl_file_expressions(included))
    end
    return exs
end

# Returns `file => name` for each use of `Reactant`, or of a name that only
# a Reactant file defines, in a native file of `files` (in `dir`).
function _ppl_native_reactant_uses(dir, files)
    reactant, native = Set{Symbol}([:Reactant]), Set{Symbol}()
    for file in files, ex in _ppl_file_expressions(joinpath(dir, file))
        union!(_ppl_reactant_file(file) ? reactant : native, _ppl_defined_names(ex))
    end
    setdiff!(reactant, native)
    uses = Pair{String,Symbol}[]
    for file in files
        _ppl_reactant_file(file) && continue
        symbols = Set{Symbol}()
        foreach(ex -> _ppl_symbols!(symbols, ex), _ppl_file_expressions(joinpath(dir, file)))
        append!(uses, (file => name for name in sort!(collect(intersect(symbols, reactant)))))
    end
    return uses
end
