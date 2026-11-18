# Where a prepared program's time goes: statement scopes in compiled programs
# and a profile of the value and its gradient that attributes time to them.
#
# Every program RK compiles (`_compile`, codegen.jl) runs each statement that
# assigns a value inside a source scope named `rk:<name>`, `<name>` being the
# assigned local without its generated numbering (`mu`, `_ppl_lik_y_resp`,
# `plate_heavy` for a plate cell's `heavy`). A scope is the lowered form Julia
# gives a macro expansion: `Expr(:meta, :push_loc, file)` before the statement
# and `Expr(:meta, :pop_loc)` after it. It is debug information only, so the
# compiled code is the same. (A bare `LineNumberNode` cannot carry the name:
# lowering keeps only line nodes of the method's own file.) The whole body runs
# in a scope `rk:(other statements)`, so code between statements (loop control,
# branch conditions, calls whose result is not assigned) has a name too.
# Julia keeps scopes through inlining, so the frames of a sample in a plate
# loop, a cell kernel or an inlined record name the statement the sample ran.
# Enzyme keeps the debug locations of the instructions it differentiates, so
# the frames of a sample in the reverse pass name the same statement, which is
# what lets `ad_cost_profile` split the actual (fused) gradient by statement.
#
# Readers of a compiled program's source (`_program_expression`) get it without
# the scopes: displays, the JAX emitter and the AD slot analysis see what they
# saw before the scopes existed.

const _SOURCE_MARKER_PREFIX = "rk:"
const _SOURCE_MARKER_PROGRAM = "(other statements)"

_source_scope(name::AbstractString) =
    Expr(:meta, :push_loc, Symbol(_SOURCE_MARKER_PREFIX, name))
const _SOURCE_SCOPE_END = Expr(:meta, :pop_loc)

_is_source_scope(x) = x isa Expr && x.head === :meta && length(x.args) == 2 &&
    x.args[1] === :push_loc && x.args[2] isa Symbol &&
    startswith(String(x.args[2]), _SOURCE_MARKER_PREFIX)
_is_scope_end(x) = x isa Expr && x.head === :meta && length(x.args) == 1 &&
    x.args[1] === :pop_loc

# The name a scope shows for an assigned local: generated numbering and the
# leading `#`s of a generated name dropped.
function _source_marker_label(name::Symbol)
    text = lstrip(replace(String(name), r"#\d+" => ""), '#')
    isempty(text) ? nothing : String(text)
end

# Heads of statements that assign: `=` and Julia's updating forms.
const _SOURCE_MARKER_ASSIGNMENTS = (:(=), :.=, :+=, :-=, :*=, :/=, :.+=, :.-=,
    :.*=, :./=, :^=, :÷=, :%=, :|=, :&=, :⊻=, :<<=, :>>=, :>>>=, :\=)

function _source_marker_name(lhs)
    lhs isa Symbol && return _source_marker_label(lhs)
    lhs isa Expr || return nothing
    lhs.head in (:(::), :ref, :.) && return _source_marker_name(lhs.args[1])
    if lhs.head === :tuple
        names = String[]
        for arg in lhs.args
            name = _source_marker_name(arg)
            name === nothing || push!(names, name)
        end
        return isempty(names) ? nothing : join(names, ", ")
    end
    nothing
end

# The scope name of a block statement, or `nothing` for a statement that
# assigns nothing (it runs in the enclosing scope).
function _statement_marker_name(statement)
    statement isa Expr || return nothing
    head = statement.head
    head in (:local, :global) && length(statement.args) == 1 &&
        return _statement_marker_name(statement.args[1])
    head in _SOURCE_MARKER_ASSIGNMENTS && length(statement.args) == 2 &&
        return _source_marker_name(statement.args[1])
    nothing
end

# The program with statement scopes. Idempotent: scopes already present are
# removed first.
function _source_markers(ast::Expr)
    ast = _strip_source_markers(ast)
    ast.head in (:function, :->) && length(ast.args) == 2 || return ast
    body = _insert_source_markers(ast.args[2], true)
    # Julia's own generated-function body has this shape: the scope, then a
    # return of the body, so the body's value is unchanged.
    Expr(ast.head, ast.args[1], Expr(:block, _source_scope(_SOURCE_MARKER_PROGRAM),
        Expr(:return, body), _SOURCE_SCOPE_END))
end

# `value` says whether the expression's value is used. A block whose value is
# used keeps its last statement unwrapped (a scope's end would replace the
# block's value); a statement before the last, a loop body and the branches of
# an `if` whose value is not used are statements whose value is discarded.
_insert_source_markers(x, value::Bool = true) = x
function _insert_source_markers(ex::Expr, value::Bool = true)
    head = ex.head
    head in (:quote, :inert, :meta, :line) && return ex
    mark(x) = _insert_source_markers(x, true)
    # Binding lists, iteration specifications and signatures are not
    # statement blocks.
    if head === :let && length(ex.args) == 2
        return Expr(:let, ex.args[1], _insert_source_markers(ex.args[2], value))
    elseif head in (:for, :while) && length(ex.args) == 2
        # A loop's body has no value; a `for` loop's iteration specification
        # is a block of assignments that must stay as it is.
        return Expr(head, head === :for ? ex.args[1] : mark(ex.args[1]),
                    _insert_source_markers(ex.args[2], false))
    elseif head in (:if, :elseif) && length(ex.args) in (2, 3)
        return Expr(head, mark(ex.args[1]),
                    Any[_insert_source_markers(arm, value) for arm in ex.args[2:end]]...)
    elseif head in (:function, :->) && length(ex.args) == 2
        return Expr(head, ex.args[1], mark(ex.args[2]))
    elseif head === :macrocall
        return Expr(:macrocall, ex.args[1:min(end, 2)]...,
                    Any[_insert_source_markers(arg, value) for arg in ex.args[3:end]]...)
    elseif head in _SOURCE_MARKER_ASSIGNMENTS && length(ex.args) == 2
        return Expr(head, ex.args[1], mark(ex.args[2]))
    elseif head === :block
        last_value = findlast(arg -> !(arg isa LineNumberNode), ex.args)
        args = Any[]
        for (index, statement) in enumerate(ex.args)
            used = value && index === last_value
            marked = _insert_source_markers(statement, used)
            name = used ? nothing : _statement_marker_name(statement)
            if name === nothing
                push!(args, marked)
            else
                push!(args, _source_scope(name), marked, _SOURCE_SCOPE_END)
            end
        end
        return Expr(:block, args...)
    end
    Expr(head, Any[mark(arg) for arg in ex.args]...)
end

_strip_source_markers(x) = x
function _strip_source_markers(ex::Expr)
    ex.head === :quote && return ex
    _has_source_markers(ex) || return ex
    # The program scope `_source_markers` wraps a function body in.
    if ex.head in (:function, :->) && length(ex.args) == 2
        body = ex.args[2]
        if body isa Expr && body.head === :block && length(body.args) == 3 &&
                _is_source_scope(body.args[1]) && _is_scope_end(body.args[3]) &&
                body.args[2] isa Expr && body.args[2].head === :return &&
                length(body.args[2].args) == 1
            return Expr(ex.head, ex.args[1], _strip_source_markers(body.args[2].args[1]))
        end
    end
    if ex.head === :block
        args = Any[]
        index = 1
        while index <= length(ex.args)
            arg = ex.args[index]
            if _is_source_scope(arg) && index + 2 <= length(ex.args) &&
                    _is_scope_end(ex.args[index + 2])
                push!(args, _strip_source_markers(ex.args[index + 1]))
                index += 3
            else
                push!(args, _strip_source_markers(arg))
                index += 1
            end
        end
        return Expr(:block, args...)
    end
    Expr(ex.head, Any[_strip_source_markers(arg) for arg in ex.args]...)
end
_has_source_markers(x) = _is_source_scope(x)
_has_source_markers(ex::Expr) = _is_source_scope(ex) ||
    (ex.head !== :quote && any(_has_source_markers, ex.args))

# --- profiling ---------------------------------------------------------------

"""
    ADCostPart

One row of an [`ADCostProfile`](@ref): a statement of the profiled program, or
the time spent outside RK's generated code. `name` is the statement's assigned
value (`mu`, `_ppl_lik_y_resp`, `plate_heavy` for the statement `heavy` of a
plate cell); statements of one name are counted together. Names in
parentheses group the other samples: `(other statements)` (code of an RK
program between its assigning statements: loop control, branch conditions,
calls whose result is not assigned; code of a program RK compiled for a
recipe, such as a derivative rule's, counts towards the statement that calls
it instead), `(outside generated code)` (the AD
driver, DifferentiationInterface, Enzyme's runtime and the caller) and
`(garbage collection)` (every sample inside a collection, whichever statement
allocated when it started).

- `primal_seconds`, `gradient_seconds`: the part's estimated time per call,
  its share of the samples times the measured time per call.
- `primal_samples`, `gradient_samples`: the samples the estimate rests on.
- `primal_bytes`, `gradient_bytes`: Julia heap bytes it allocated per warm
  call (`NaN` when allocations were not profiled).
- `primal_memory_seconds`, `gradient_memory_seconds`: the part of that time
  spent allocating, freeing, zero-filling or copying memory in Julia's runtime
  or the C library: Julia heap allocation, and in the gradient also the shadows
  and the native `malloc`/`free` of the reverse pass's tape, which native
  Enzyme books on the statement whose value it stores or on the loop it
  stores it in.
- `callees`: the functions the part's time is spent in, when the statement
  calls a function that is neither Julia's nor RK's own (a model's helper, a
  package function), each as `(; name, primal_seconds, gradient_seconds)`,
  `name` being the function and its file, sorted by gradient time. The
  outermost such call counts, so a helper that calls further helpers is named
  itself.
"""
struct ADCostPart
    name::String
    primal_seconds::Float64
    gradient_seconds::Float64
    primal_samples::Int
    gradient_samples::Int
    primal_bytes::Float64
    gradient_bytes::Float64
    primal_memory_seconds::Float64
    gradient_memory_seconds::Float64
    callees::Vector{NamedTuple{(:name, :primal_seconds, :gradient_seconds),
                               Tuple{String,Float64,Float64}}}
end

"""
    ADCostProfile

What [`ad_cost_profile`](@ref) measured: the whole program's warm time per
call of the value (`primal_seconds`) and of the value with its gradient
(`gradient_seconds`), their `ratio`, the Julia heap bytes of one warm call of
each (`primal_bytes`, `gradient_bytes`), and, when the program was profiled
(`profiled`), its `parts` sorted by gradient time. `threshold` is the ratio a
profile was conditional on (`nothing`: always profiled). `primal_samples` and
`gradient_samples` count the profile samples the parts' estimates rest on.
"""
struct ADCostProfile
    primal_seconds::Float64
    gradient_seconds::Float64
    ratio::Float64
    primal_bytes::Int
    gradient_bytes::Int
    threshold::Union{Nothing,Float64}
    profiled::Bool
    primal_samples::Int
    gradient_samples::Int
    parts::Vector{ADCostPart}
end

"""
    ad_cost_profile(value, value_and_gradient; seconds = 1.0, threshold = nothing,
                    allocations = true) -> ADCostProfile
    ad_cost_profile(prepared::PreparedADKernel, args...; kernel = prepared.kernel, kwargs...)

Measure where a prepared program's value and gradient spend their time.
`value` and `value_and_gradient` are functions of no arguments that evaluate
the program once (a value, then a value with its gradient, as a sampler calls
them). The second form profiles `kernel(args...)` against
`ad_value_and_gradient(prepared, args...)`.

It first times warm calls of both and reports the ratio of the gradient's time
to the value's. With `threshold`, it stops there unless the ratio exceeds
`threshold` (`4.0` is a usual bound for a reverse gradient), so a benchmark
can call it on every program. Otherwise it runs each function for `seconds`
under Julia's sampling profiler and attributes every sample of the calling
thread to the RK statement it ran: every program RK compiles runs each
statement that assigns a value in a source scope with its name, as Julia does
for a macro expansion. Native Enzyme keeps those scopes in the reverse pass,
so a statement's gradient time includes its reverse code inside the one fused
derivative, and the parts add up to the whole. With
`allocations`, a few further calls run under `Profile.Allocs` to attribute
the Julia heap bytes of a warm call the same way.

The result lists parts by gradient time, each with its gradient ÷ value ratio.
Estimates rest on the sample counts shown; a part with few samples is
imprecise. Profiling clears Julia's profile buffer and restores the
profiler's settings afterwards. Measure on a quiet thread: the samples of
other threads are ignored, but their load slows the measured one.

Time spent managing memory is reported beside the statements, because a
gradient's excess over its value is often there rather than in arithmetic:
each part's `memory` share is its time allocating, freeing, zero-filling or
copying (Julia heap allocations and their shadows, and native Enzyme's tape,
which it allocates with `malloc` and frees after the reverse pass), and every
sample inside a garbage collection is counted as `(garbage collection)`
instead of on the statement whose allocation happened to start it, which
varies from run to run and host to host. The display's first lines give both
shares for the whole value and gradient.

Statements in a function Julia does not inline and RK did not compile (a
package function a recipe calls) count towards the RK statement that calls it,
and that part's `callees` names the function (the display's `mostly in`
column shows the largest when it holds at least half of the part's gradient
time). Native Enzyme tapes are not Julia heap bytes; their
`malloc` time is in `memory`, and `Enzyme.API.printperf!(true)` on the part
this profile names shows what is stored.
"""
function ad_cost_profile(value, value_and_gradient; seconds::Real = 1.0,
                         threshold::Union{Nothing,Real} = nothing,
                         allocations::Bool = true)
    seconds > 0 || throw(ArgumentError("`seconds` must be positive; got $seconds"))
    # Warm: compile both, then run each once more.
    value(); value_and_gradient()
    value(); value_and_gradient()
    budget = max(seconds / 4, 0.05)
    primal = _seconds_per_call(value, budget)
    gradient = _seconds_per_call(value_and_gradient, budget)
    primal_bytes = _allocated_per_call(value)
    gradient_bytes = _allocated_per_call(value_and_gradient)
    ratio = gradient / primal
    limit = threshold === nothing ? nothing : Float64(threshold)
    if limit !== nothing && !(ratio > limit)
        return ADCostProfile(primal, gradient, ratio, primal_bytes, gradient_bytes,
                             limit, false, 0, 0, ADCostPart[])
    end
    primal_samples = _marker_samples(value, seconds)
    gradient_samples = _marker_samples(value_and_gradient, seconds)
    primal_total, gradient_total = primal_samples.total, gradient_samples.total
    primal_allocated = allocations ? _marker_allocations(value) : Dict{String,Float64}()
    gradient_allocated = allocations ? _marker_allocations(value_and_gradient) :
        Dict{String,Float64}()
    names = union(keys(primal_samples.counts), keys(gradient_samples.counts),
                  keys(primal_allocated), keys(gradient_allocated))
    share(counts, total, key) = total == 0 ? 0.0 : get(counts, key, 0) / total
    bytes(allocated, name) = allocations ? get(allocated, name, 0.0) : NaN
    function callees(name)
        called = union((callee for (label, callee) in keys(primal_samples.callees) if label == name),
                       (callee for (label, callee) in keys(gradient_samples.callees) if label == name))
        rows = [(; name = callee,
                   primal_seconds = primal * share(primal_samples.callees, primal_total, (name, callee)),
                   gradient_seconds = gradient * share(gradient_samples.callees, gradient_total, (name, callee)))
                for callee in called]
        sort!(rows; by = row -> (-row.gradient_seconds, -row.primal_seconds, row.name))
    end
    parts = ADCostPart[ADCostPart(name,
        primal * share(primal_samples.counts, primal_total, name),
        gradient * share(gradient_samples.counts, gradient_total, name),
        get(primal_samples.counts, name, 0), get(gradient_samples.counts, name, 0),
        bytes(primal_allocated, name), bytes(gradient_allocated, name),
        primal * share(primal_samples.memory, primal_total, name),
        gradient * share(gradient_samples.memory, gradient_total, name),
        callees(name))
        for name in names]
    sort!(parts; by = part -> (-part.gradient_seconds, -part.primal_seconds, part.name))
    ADCostProfile(primal, gradient, ratio, primal_bytes, gradient_bytes, limit, true,
                  primal_total, gradient_total, parts)
end

function ad_cost_profile(prepared::PreparedADKernel, args...;
                         kernel = prepared.kernel, kwargs...)
    ad_cost_profile(() -> kernel(args...),
                    () -> ad_value_and_gradient(prepared, args...); kwargs...)
end

# Median seconds per call over five batches of about `budget / 5` seconds each.
function _seconds_per_call(f::F, budget::Real) where {F}
    n = 1
    while true
        elapsed = _batch_seconds(f, n)
        elapsed >= budget / 5 && break
        n = elapsed <= 0 ? 10n : max(n + 1, ceil(Int, n * budget / 5 / elapsed))
    end
    per_call = sort!([_batch_seconds(f, n) / n for _ in 1:5])
    per_call[3]
end
function _batch_seconds(f::F, n::Int) where {F}
    start = time_ns()
    for _ in 1:n
        f()
    end
    (time_ns() - start) / 1e9
end
_allocated_per_call(f::F) where {F} = (f(); @allocated f())

# The label of a profile sample: `(garbage collection)` for a sample inside a
# collection, else its innermost RK statement marker, else the innermost
# program scope `(other statements)`, else `(outside generated code)`.
# `frames` run from the innermost frame outwards.
_sample_label(frames) = first(_sample_info(collect(frames)))

# A sample's label, whether it was spent managing memory, and the function it
# was spent in when its statement called one that is neither Julia's nor RK's
# (`nothing` otherwise). A program's own scope `(other statements)` takes a
# sample only when no statement encloses it: a program RK compiles for a
# recipe's expression (a rule's `q = exp(-abs(x))`) runs inside the statement
# that calls the rule, so its time is that statement's.
function _sample_info(frames::AbstractVector)
    any(_collection_frame, frames) && return ("(garbage collection)", false, nothing)
    memory = _memory_sample(frames)
    program = 0
    for (index, frame) in enumerate(frames)
        label = _marker_label(frame)
        label === nothing && continue
        if label == _SOURCE_MARKER_PROGRAM
            program == 0 && (program = index)
            continue
        end
        return (label, memory, _sample_callee(frames, index))
    end
    program == 0 || return (_SOURCE_MARKER_PROGRAM, memory, _sample_callee(frames, program))
    ("(outside generated code)", memory, nothing)
end

function _marker_label(frame)
    file = String(frame.file)
    startswith(file, "./") && (file = file[3:end])
    startswith(file, _SOURCE_MARKER_PREFIX) || return nothing
    file[(ncodeunits(_SOURCE_MARKER_PREFIX) + 1):end]
end

# Julia's collector: its entry points and the mark and sweep phases they run.
function _collection_frame(frame)
    name = String(frame.func)
    name in ("_jl_gc_collect", "jl_gc_collect", "ijl_gc_collect", "gc") &&
        (name != "gc" || endswith(String(frame.file), "gcutils.jl")) && return true
    startswith(name, "gc_mark") || startswith(name, "gc_sweep")
end

# A sample is memory management when the native code it ran (the frames
# before the first Julia frame) allocates, frees, zero-fills or copies: Julia's
# allocator, or the C library's (`malloc`, `free` and their internals,
# `memset`, `memcpy`), whose internal functions often have no symbol name.
const _MEMORY_FUNCTION = r"^(i?jl_(gc_(pool_|big_|small_|managed_)?(alloc|malloc|realloc)|alloc_|malloc|realloc|calloc|free|array_grow|array_del)|_new_array|_new_genericmemory|gc_(alloc|add_page|page_data)|set_page_metadata|_?_?(libc_)?(malloc|calloc|realloc|free|memalign|posix_memalign|aligned_alloc)|_int_|tcache|mem(set|cpy|move)|__mem(set|cpy|move)|_platform_mem|bzero|__bzero|szone_|tiny_|small_|medium_|nanov2_|_malloc_|_free)"
const _MEMORY_LIBRARY = r"(^|/)(libc\.so|libc-|libsystem_malloc|libsystem_platform)"
function _memory_sample(frames)
    for frame in frames
        _native_frame(frame) || return false
        name = String(frame.func)
        occursin(_MEMORY_FUNCTION, name) && return true
        isempty(name) && occursin(_MEMORY_LIBRARY, String(frame.file)) && return true
    end
    false
end
function _native_frame(frame)
    frame.from_c && return true
    file = String(frame.file)
    endswith(file, ".c") || endswith(file, ".h") || endswith(file, ".cpp") ||
        occursin(".so", file) || occursin(".dylib", file)
end

# The outermost function, inside the statement at `frames[marker]`, that is
# neither Julia's (Base, Core, a standard library) nor RK's own source.
function _sample_callee(frames, marker::Int)
    for index in (marker - 1):-1:1
        frame = frames[index]
        _native_frame(frame) && continue
        _marker_label(frame) === nothing || continue
        name = String(frame.func)
        (isempty(name) || name == "macro expansion") && continue
        file = String(frame.file)
        _language_file(file) && continue
        return string(name, " (", basename(file), ")")
    end
    nothing
end
# RK's own source: the package and its extensions.
const _RK_SOURCE = dirname(@__FILE__)
const _RK_OWN_DIRECTORIES = (joinpath(_RK_SOURCE, ""), joinpath(dirname(_RK_SOURCE), "ext", ""))
function _language_file(file)
    # Base's files are recorded relative to its source directory (`./array.jl`),
    # generated code as `none`.
    isabspath(file) || return true
    (occursin("/share/julia/base/", file) || occursin("/share/julia/stdlib/", file)) &&
        return true
    any(directory -> startswith(file, directory), _RK_OWN_DIRECTORIES)
end

# Samples of `f` run for `seconds` on this thread, counted by label (`counts`),
# by label for those spent managing memory (`memory`) and by label and called
# function (`callees`).
function _marker_samples(f::F, seconds::Real) where {F}
    previous = Profile.init()
    thread = Threads.threadid()
    counts = Dict{String,Int}()
    memory = Dict{String,Int}()
    callees = Dict{Tuple{String,String},Int}()
    total = 0
    try
        # About 10^4 samples per second, with room for deep stacks.
        Profile.init(; n = max(first(previous), 10^7), delay = 1e-4)
        Profile.clear()
        deadline = time() + seconds
        Profile.@profile while time() < deadline
            f()
        end
        data = Profile.fetch(; include_meta = true)
        frames = Profile.getdict(data)
        start = 1
        for i in eachindex(data)
            Profile.is_block_end(data, i) || continue
            ips = view(data, start:(i - Profile.nmeta - 2))
            start = i + 1
            data[i - Profile.META_OFFSET_THREADID] == thread || continue
            # The sleep state is stored plus one: 1 running, 2 sleeping.
            data[i - Profile.META_OFFSET_SLEEPSTATE] == 1 || continue
            label, managing, callee = _sample_info(collect(Iterators.flatten(
                get(frames, ip, Base.StackTraces.StackFrame[]) for ip in ips)))
            counts[label] = get(counts, label, 0) + 1
            managing && (memory[label] = get(memory, label, 0) + 1)
            callee === nothing ||
                (callees[(label, callee)] = get(callees, (label, callee), 0) + 1)
            total += 1
        end
    finally
        Profile.clear()
        Profile.init(first(previous), last(previous))
    end
    (; counts, memory, callees, total)
end

# Julia heap bytes per warm call of `f`, by label, from three calls under
# `Profile.Allocs` with every allocation recorded.
function _marker_allocations(f::F; calls::Int = 3) where {F}
    f()
    Profile.Allocs.clear()
    bytes = Dict{String,Float64}()
    try
        Profile.Allocs.@profile sample_rate = 1.0 for _ in 1:calls
            f()
        end
        for allocation in Profile.Allocs.fetch().allocs
            label = _sample_label(allocation.stacktrace)
            bytes[label] = get(bytes, label, 0.0) + allocation.size / calls
        end
    finally
        Profile.Allocs.clear()
    end
    bytes
end

# --- display -----------------------------------------------------------------

function _cost_time(seconds::Real)
    seconds >= 1 && return string(round(seconds; digits = 3), " s")
    seconds >= 1e-3 && return string(round(seconds * 1e3; digits = 3), " ms")
    string(round(seconds * 1e6; digits = 2), " µs")
end
function _cost_bytes(bytes::Real)
    isnan(bytes) && return "—"
    bytes < 1024 && return string(round(Int, bytes), " B")
    bytes < 1024^2 && return string(round(bytes / 1024; digits = 1), " KiB")
    string(round(bytes / 1024^2; digits = 1), " MiB")
end
_cost_ratio(gradient, primal) = primal > 0 ? string(round(gradient / primal; digits = 2), "×") : "—"
_cost_share(part, whole) = whole > 0 ? string(round(Int, 100 * part / whole), "%") : "—"
# The function a part's gradient time is mostly spent in, with its share: shown
# only when it holds at least half of that time (`callees` lists every one).
function _cost_callee(part::ADCostPart)
    isempty(part.callees) && return ""
    top = first(part.callees)
    top.gradient_seconds >= 0.5 * part.gradient_seconds > 0 || return ""
    string(top.name, " ", _cost_share(top.gradient_seconds, part.gradient_seconds))
end
# A pass's share of time managing memory, then its share in garbage collection.
function _cost_memory_summary(p::ADCostProfile, pass::Symbol)
    total = getfield(p, Symbol(pass, :_seconds))
    memory = sum(part -> getfield(part, Symbol(pass, :_memory_seconds)), p.parts; init = 0.0)
    collection = sum(part -> part.name == "(garbage collection)" ?
                     getfield(part, Symbol(pass, :_seconds)) : 0.0, p.parts; init = 0.0)
    string(_cost_share(memory, total), " + garbage collection ", _cost_share(collection, total))
end

function Base.show(io::IO, ::MIME"text/plain", p::ADCostProfile)
    println(io, "ADCostProfile: gradient ", _cost_time(p.gradient_seconds), " = ",
            round(p.ratio; digits = 2), "× value ", _cost_time(p.primal_seconds),
            " per call; heap ", _cost_bytes(p.gradient_bytes), " / ",
            _cost_bytes(p.primal_bytes))
    if !p.profiled
        print(io, "not profiled: the ratio does not exceed the threshold ", p.threshold)
        return
    end
    println(io, "samples: ", p.gradient_samples, " gradient, ", p.primal_samples, " value")
    println(io, "memory management: gradient ", _cost_memory_summary(p, :gradient),
            "; value ", _cost_memory_summary(p, :primal))
    # Parts with no sample and no allocation in either pass are left out of
    # the table, and at most `_COST_ROWS` rows are shown; `p.parts` has all.
    shown = [part for part in p.parts if part.gradient_samples + part.primal_samples > 0 ||
             (!isnan(part.gradient_bytes) && part.gradient_bytes + part.primal_bytes > 0)]
    hidden = length(p.parts) - min(length(shown), _COST_ROWS)
    rows = [(part.name, _cost_time(part.gradient_seconds), _cost_time(part.primal_seconds),
             _cost_ratio(part.gradient_seconds, part.primal_seconds),
             _cost_share(part.gradient_memory_seconds, part.gradient_seconds),
             string(part.gradient_samples), _cost_bytes(part.gradient_bytes),
             _cost_callee(part))
            for part in first(shown, _COST_ROWS)]
    header = ("part", "gradient", "value", "ratio", "memory", "samples", "gradient heap",
              "mostly in")
    n = length(header)
    widths = [maximum(length, (header[k], (row[k] for row in rows)...); init = 0) for k in 1:n]
    line(row) = rstrip(join((k == 1 || k == n ? rpad(row[k], widths[k]) : lpad(row[k], widths[k])
                             for k in 1:n), "  "))
    print(io, line(header))
    for row in rows
        print(io, '\n', line(row))
    end
    hidden > 0 && print(io, "\n… ", hidden, " more parts with few or no samples (`parts`)")
end
const _COST_ROWS = 20
Base.show(io::IO, p::ADCostProfile) = print(io, "ADCostProfile(gradient ",
    _cost_time(p.gradient_seconds), ", ", round(p.ratio; digits = 2), "× value, ",
    length(p.parts), " parts)")
