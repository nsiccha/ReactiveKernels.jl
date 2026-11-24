# The check of the derivative native Enzyme compiles for a prepared program.
#
# `type_stability` checks the primal Julia code of a prepared program. With
# native Enzyme, a gradient runs a different program: the derivative Enzyme
# compiles from that code, a forward sweep that records a tape and a reverse
# sweep. Enzyme's dogfood line reports what that compiled code calls at run
# time (`Enzyme.derivative_report`, dogfood repair `derivative_report`, Enzyme
# snag `no-supported-way-c302943c`): calls dispatched at run time, calls Enzyme
# differentiates at run time, boxes, and allocations on Julia's heap or native
# ones (Enzyme's tape). RK reads that report (snag `rk-ad-pass-type-b179e693`,
# user decision
# `ReactiveKernels:snag.rk-ad-pass-type-b179e693/decisions/2026-11-17T22-00-23-356-1auv5ma`):
#
# - At `type_stability_level = :all`, every native-Enzyme `prepare_ad*`
#   evaluates its operator once at the exemplar, as `warm = true` does, and
#   checks the derivatives that evaluation ran. Run-time dispatch and boxes on
#   paths that return are findings of `type_stability`: warned once per
#   process, or refused with `TypeInstabilityError`. `:rk` and `:values` check
#   the primal only.
# - `ad_allocations` (`:ignore`, the default, `:warn` or `:error`) does the
#   same with the derivative's allocations, refused with
#   `DerivativeAllocationError`. A setting other than `:ignore` checks the
#   derivative at every level.
# - `ad_derivative_report` returns every finding, allocations included.
#
# Without Enzyme's report the derivative is not checked, and the preparation
# proceeds as before: a preparation checked with `type_stability = :error` or
# with `ad_allocations` warns once per process that it is not.

"""
    ADDerivativeReport

What [`ad_derivative_report`](@ref) returns: the derivatives native Enzyme
compiled, linked from a package image or ran again while the reported code
ran, with ReactiveKernels' reading of their findings.

- `value`: the result of the code that ran.
- `derivatives`: one `Enzyme.ReportedDerivative` per derivative, as Enzyme
  reports it (`enzyme-use`, "What a compiled derivative calls at run time"):
  the differentiated function's type `func`, its `mode`, its `origin`, and its
  `findings`, each an `Enzyme.DerivativeFinding` of a call that leaves the
  compiled code, or `nothing` where Enzyme did not inspect the code.
- `unstable`: `derivative => finding` pairs for the calls that make a
  derivative not type stable: a call dispatched at run time
  (`:dynamic_dispatch`), a call Enzyme differentiates at run time
  (`:enzyme_runtime_fallback`), or a box (`:box`).
- `allocations`: `derivative => finding` pairs for the derivative's
  allocations, on Julia's heap (`:gc_allocation`) or native (`:native_allocation`,
  such as Enzyme's tape), with their size where it is a constant (`bytes`).

A finding's `frames` are its source lines, innermost first. In the program
ReactiveKernels generates, the statement a call is in is a frame
`macro expansion` at line 1 of the file `./rk:<value>`: `<value>` is the value
the statement assigns, the name [`ad_cost_profile`](@ref) reports
(`plate_<value>` for a plate cell's `<value>`, on some loop paths with a
suffix such as `#direct`), and `(other statements)` is code between
statements (loop control, branch conditions, calls whose result is not
assigned), such as the buffers Enzyme keeps for a plate's loop as a whole. A
statement that assigns inside another is a frame inside the other's. Calls of
one function in different statements are different findings, each with its
own `count`. A finding inside a function Enzyme
differentiates on its own (an `in_function` such as `augmented_julia_copyto_…`)
has frames up to its call site only where that function has one call site in
the derivative's code; with several, its frames end in the function itself
(Enzyme snag `derivative-repor-e519a164`).

`unstable` and `allocations` leave out calls only reached on a path that ends
in an error (`throw_only`), and calls an expansion of Julia's logging macros
writes, which the primal check exempts too. Calls in code the derivative runs
without differentiating it (`sweep === :primal`) count in `unstable` where the
loaded Enzyme has the dogfood repair `derivative_report_reachability`, which
marks the code of a function only called on an error path, such as one
building an error message, `throw_only` (Enzyme snag
`derivative-repor-8d7dfce1`), unless they sit in Julia's own functions (Base,
Core, the standard libraries), whose bodies the primal check treats as leaves
too. Without that repair `unstable` leaves undifferentiated code to the primal
check, since error-only code would read as returning. Every other call that
leaves the compiled code
(`:runtime_call`: BLAS, Julia's runtime, separately compiled Julia code) is
in `derivatives` only.
"""
struct ADDerivativeReport
    value::Any
    derivatives::Vector{Any}
    unstable::Vector{Pair{Any,Any}}
    allocations::Vector{Pair{Any,Any}}
end

const _DERIVATIVE_UNSTABLE_KINDS = (:dynamic_dispatch, :enzyme_runtime_fallback, :box)
const _DERIVATIVE_ALLOCATION_KINDS = (:gc_allocation, :native_allocation)

_reported_findings(derivative) = something(derivative.findings, ())

# A call written by an expansion of Julia's logging macros: Julia's own code,
# which dispatches at run time by design and runs only for a record a logger
# accepts. The primal audit exempts it (`_code_logging`), and so does this one.
_derivative_logging(finding) = any(finding.frames) do frame
    frame.func === Symbol("macro expansion") || return false
    # Debug info names Base's file `./logging.jl` where its methods say
    # `logging.jl`.
    file = chopprefix(String(frame.file), "./")
    any(source -> file == chopprefix(String(source), "./"), _LOGGING_SOURCES)
end

# Whether the loaded Enzyme marks the code of a function only called on an
# error path `throw_only` (repair `derivative_report_reachability`), so that
# `unstable` can count code the derivative runs without differentiating it.
# Set by the Enzyme extension when it loads.
const _DERIVATIVE_REPORT_REACHABILITY = Ref(false)

# Whether a frame of debug info is in Julia's own code (Base, Core and the
# standard libraries), whose bodies the primal check treats as leaves: debug
# info names Base's files `./x.jl` and the standard libraries by their install
# path. A statement scope of RK's generated code (`./rk:<name>`, ad_cost.jl)
# is not.
function _julia_own_frame(frame)
    _marker_label(frame) === nothing || return false
    file = String(frame.file)
    startswith(file, "./") || startswith(file, Sys.STDLIB) ||
        occursin("/share/julia/base/", file) || occursin("/share/julia/stdlib/", file)
end

# Whether a finding counts as unstable: code the derivative differentiates
# always does; code it runs undifferentiated does where Enzyme marks error-only
# code (`derivative_report_reachability`), except inside Julia's own functions,
# which the primal check exempts too (their logging machinery included).
_derivative_unstable_sweep(finding, primal::Bool) =
    finding.sweep !== :primal ||
    primal && (isempty(finding.frames) || !_julia_own_frame(first(finding.frames)))

function ADDerivativeReport(value, derivatives)
    primal = _DERIVATIVE_REPORT_REACHABILITY[]
    unstable = Pair{Any,Any}[]
    allocations = Pair{Any,Any}[]
    for derivative in derivatives, finding in _reported_findings(derivative)
        (finding.throw_only || _derivative_logging(finding)) && continue
        finding.kind in _DERIVATIVE_UNSTABLE_KINDS &&
            _derivative_unstable_sweep(finding, primal) &&
            push!(unstable, derivative => finding)
        finding.kind in _DERIVATIVE_ALLOCATION_KINDS &&
            push!(allocations, derivative => finding)
    end
    ADDerivativeReport(value, collect(Any, derivatives), unstable, allocations)
end

function Base.show(io::IO, report::ADDerivativeReport)
    n = length(report.derivatives)
    print(io, "ADDerivativeReport(", n, n == 1 ? " derivative, " : " derivatives, ",
          length(report.unstable), " unstable, ", length(report.allocations),
          " allocations)")
end

function Base.show(io::IO, ::MIME"text/plain", report::ADDerivativeReport)
    show(io, report)
    for derivative in report.derivatives
        print(io, "\n")
        show(io, MIME"text/plain"(), derivative)
    end
end

# `f -> ADDerivativeReport` over `Enzyme.derivative_report`, set by the Enzyme
# extension when it loads: the function where the loaded Enzyme carries the
# repair `derivative_report`, `missing` where it does not, and `nothing` while
# Enzyme is not loaded.
const _DERIVATIVE_REPORTER = Ref{Any}(nothing)

const _DERIVATIVE_REPORT_MISSING =
    "the loaded Enzyme cannot report the derivatives it compiles. That needs " *
    "`Enzyme.derivative_report`, the Enzyme dogfood repair `derivative_report` " *
    "(dogfood 83ff174a or later; enzyme-use, \"What a compiled derivative calls " *
    "at run time\"). Registered Enzyme does not carry it."

"""
    ad_derivative_report(f::Function) -> ADDerivativeReport
    ad_derivative_report(prepared, args...; kwargs...) -> ADDerivativeReport

Run `f()` and report the derivatives native Enzyme compiled, linked from a
package image or ran again for it (see [`ADDerivativeReport`](@ref)). `f` may
evaluate any prepared AD operator, or take derivatives otherwise.

The second form evaluates a prepared operator once at `args`, as its value
function does: [`ad_value_and_gradient`](@ref) for a gradient,
[`ad_value_and_pullback`](@ref) (`args` starting with the seed),
[`ad_value_and_pushforward`](@ref), [`ad_gradient_and_hvp`](@ref) or
[`ad_gradient_hvp_and_third`](@ref) (`args` starting with the tangents). Each
derivative appears whenever it was compiled: a derivative compiled before the
report, at preparation or at an earlier call, has the findings Enzyme read when
it compiled it.

It needs Enzyme loaded, with the dogfood repair `derivative_report` (Enzyme
dogfood 83ff174a or later); otherwise it throws an `ArgumentError`. An empty
`derivatives` means the code asked native Enzyme for no derivative, not that
its derivatives are clean.

At `type_stability_level = :all` (the default), every native-Enzyme
`prepare_ad*` reads the same report for one evaluation at its exemplar and
warns about or refuses what it finds (`type_stability`, `ad_allocations`; see
[`prepare_ad`](@ref)).
"""
function ad_derivative_report(f::Function)
    reporter = _DERIVATIVE_REPORTER[]
    reporter === nothing && throw(ArgumentError(
        "`ad_derivative_report` reads native Enzyme's report of the derivatives " *
        "it compiles: load Enzyme (`using Enzyme`), with the dogfood repair " *
        "`derivative_report`."))
    reporter === missing && throw(ArgumentError(
        "`ad_derivative_report`: " * _DERIVATIVE_REPORT_MISSING))
    reporter(f)::ADDerivativeReport
end

ad_derivative_report(prepared::PreparedADKernel, args...; kwargs...) =
    ad_derivative_report(() -> ad_value_and_gradient(prepared, args...; kwargs...))
ad_derivative_report(prepared::PreparedADPullback, args...; kwargs...) =
    ad_derivative_report(() -> ad_value_and_pullback(prepared, args...; kwargs...))
ad_derivative_report(prepared::PreparedADPushforward, args...; kwargs...) =
    ad_derivative_report(() -> ad_value_and_pushforward(prepared, args...; kwargs...))
ad_derivative_report(prepared::PreparedADHVP, args...; kwargs...) =
    ad_derivative_report(() -> ad_gradient_and_hvp(prepared, args...; kwargs...))
ad_derivative_report(prepared::PreparedADThird, args...; kwargs...) =
    ad_derivative_report(() -> ad_gradient_hvp_and_third(prepared, args...; kwargs...))

# Whether a backend differentiates with native Enzyme, whose derivative the
# check reads. A second-order backend does when either part does.
_ad_native_enzyme(backend) = false
_ad_native_enzyme(::DifferentiationInterface.AutoEnzyme) = true
_ad_native_enzyme(backend::DifferentiationInterface.SecondOrder) =
    _ad_native_enzyme(DifferentiationInterface.outer(backend)) ||
    _ad_native_enzyme(DifferentiationInterface.inner(backend))

# One evaluation of a prepared operator at the exemplar `point`, `seeds` and
# `contexts`, as a function and its arguments: the operator's value function
# on a copy of the point. An array point evaluates the gradient in place.
_ad_operator_evaluation(::Val{:gradient}, prepared, point::AbstractArray, seeds, contexts) =
    _ad_prepared_value_and_gradient!, (prepared, similar(point), copy(point), contexts)
_ad_operator_evaluation(::Val{:gradient}, prepared, point, seeds, contexts) =
    _ad_prepared_value_and_gradient, (prepared, deepcopy(point), contexts)
_ad_operator_evaluation(::Val{:pullback}, prepared, point, seeds, contexts) =
    _ad_prepared_value_and_pullback,
    (prepared.call, prepared, deepcopy(point), only(seeds), contexts)
_ad_operator_evaluation(::Val{:pushforward}, prepared, point, seeds, contexts) =
    _ad_prepared_value_and_pushforward,
    (prepared.call, prepared, deepcopy(point), seeds, contexts)
_ad_operator_evaluation(::Val{:hvp}, prepared, point, seeds, contexts) =
    _ad_gradient_and_hvp, (prepared.call, prepared.preparation, prepared.backend,
                           deepcopy(point), seeds, contexts)
_ad_operator_evaluation(::Val{:third}, prepared, point, seeds, contexts) =
    _ad_gradient_hvp_and_third, (prepared, deepcopy(point), seeds, contexts)

# An evaluation that threw: the report still holds what it reached.
struct _ADEvaluationFailure
    exception::Any
end

# Check the derivative native Enzyme compiles for `prepared` with the settings
# of the preparation in progress, or warm it (`warm = true`, a gradient only)
# where nothing is checked. Returns whether it evaluated the operator.
function _ad_check_derivative(op::Val, prepared, point, seeds, contexts, warm::Bool)
    level, allocations = _type_stability_level(), _ad_allocations()
    if (level === :all || allocations !== :ignore) &&
       _ad_native_enzyme(prepared.backend) && !(prepared.call isa _ADConstantCall)
        reporter = _DERIVATIVE_REPORTER[]
        if reporter isa Function
            report = _ad_evaluation_report(reporter, op, prepared, point, seeds, contexts)
            _report_derivative(report, typeof(prepared.call),
                               level === :all ? _type_stability() : :ignore, allocations)
            return true
        end
        (_type_stability() === :error || allocations !== :ignore) &&
            @warn "ReactiveKernels does not check the derivative native Enzyme " *
                  "compiles for this preparation: " * _DERIVATIVE_REPORT_MISSING *
                  " The prepared program's own code is checked as before, and the " *
                  "preparation continues." _group = :type_stability maxlog = 1 _id =
                  :derivative_unchecked
    end
    warm && _ad_warm!(prepared, point, contexts)
    warm
end

# Evaluate the operator once inside the report. Inferring the evaluation first
# builds what the backend compiles during type inference outside the JIT
# codegen lock a first call holds (see `_ad_warm!`); `invokelatest` keeps the
# caller of the preparation from inferring it. An evaluation that throws
# leaves the rest to compile at the first call, and the preparation is kept.
function _ad_evaluation_report(reporter, op, prepared, point, seeds, contexts)
    f, args = _ad_operator_evaluation(op, prepared, point, seeds, contexts)
    Base.code_typed(f, Tuple{map(Core.Typeof, args)...})
    report = reporter() do
        try
            Base.invokelatest(f, args...)
        catch err
            err isa InterruptException && rethrow()
            _ADEvaluationFailure(err)
        end
    end
    report.value isa _ADEvaluationFailure &&
        @warn "The check of the derivative native Enzyme compiles evaluates the " *
              "prepared operator once at the exemplar, and the program threw " *
              "there. The preparation is kept: the check covers the derivatives " *
              "that evaluation reached, and whatever it did not reach compiles at " *
              "its first call." exception = report.value.exception
    report
end

"""
    DerivativeAllocationError(messages)

Thrown by a native-Enzyme `prepare_ad*` checked with `ad_allocations = :error`
when the derivative Enzyme compiles for the prepared program allocates, on
Julia's heap or natively (Enzyme's tape), on a path that returns. `messages`
holds one message per derivative, as the warning of `ad_allocations = :warn`
words it.
"""
struct DerivativeAllocationError <: Exception
    messages::Vector{String}
end

function Base.showerror(io::IO, e::DerivativeAllocationError)
    n = length(e.messages)
    print(io, "DerivativeAllocationError: the derivative native Enzyme compiles ",
          "for the prepared program allocates (", n, n == 1 ? " derivative" :
          " derivatives", "), and this preparation is checked with ",
          "`ad_allocations = :error`.")
    foreach(message -> print(io, "\n\n", message), e.messages)
end

# The findings of `pairs` (`derivative => finding`) grouped by derivative, in
# the order the report lists them.
function _findings_by_derivative(pairs)
    groups = Pair{Any,Vector{Any}}[]
    for (derivative, finding) in pairs
        index = findfirst(g -> first(g) === derivative, groups)
        index === nothing ? push!(groups, derivative => Any[finding]) :
            push!(last(groups[index]), finding)
    end
    groups
end

_derivative_mode_text(mode::Symbol) =
    mode === :reverse ? "reverse-mode" : mode === :reverse_split ?
    "split reverse-mode" : mode === :forward ? "forward-mode" : string(mode)

# The differentiated function as the author knows it: the prepared program
# itself, or a function it reaches, by name rather than by its full type.
function _derivative_description(derivative, program::Type)
    func = derivative.func
    what = func == program ? "the prepared program" :
        "`" * (func isa DataType ? string(parentmodule(func), ".", nameof(func)) :
               string(func)) * "`, which the prepared program reaches,"
    "the $(_derivative_mode_text(derivative.mode)) derivative native Enzyme " *
    "compiles for $what"
end

const _DERIVATIVE_FINDING_TEXT = Dict(
    :dynamic_dispatch => "a call dispatched at run time",
    :enzyme_runtime_fallback => "a call Enzyme differentiates at run time",
    :box => "a boxed value",
    :gc_allocation => "an allocation on Julia's heap",
    :native_allocation => "a native allocation",
)

_derivative_sweep_text(sweep::Symbol) =
    sweep === :forward ? "in the forward sweep" : sweep === :reverse ?
    "in the reverse sweep" : sweep === :combined ? "in code holding both sweeps" :
    sweep === :primal ? "in code that is not differentiated" : "in $sweep code"

function _derivative_finding_line(finding)
    details = String[]
    if finding.kind === :native_allocation
        push!(details, finding.bytes === nothing ? "of a size known only at run time" :
                       "$(finding.bytes) bytes")
    elseif finding.kind === :gc_allocation && finding.bytes !== nothing
        push!(details, "$(finding.bytes) bytes")
    end
    push!(details, _derivative_sweep_text(finding.sweep))
    finding.in_loop && push!(details, "in a loop")
    finding.count == 1 || push!(details, "$(finding.count) calls")
    head = "  - $(_DERIVATIVE_FINDING_TEXT[finding.kind]), `$(finding.callee)` (" *
           join(details, ", ") * ")"
    join([head; ["      at $(frame.func) @ $(frame.file):$(frame.line)"
                 for frame in finding.frames]], "\n")
end

function _derivative_message(::Val{:unstable}, derivative, findings, program)
    "ReactiveKernels: $(_derivative_description(derivative, program)) is not type " *
    "stable:\n" * join(map(_derivative_finding_line, findings), "\n") *
    "\nEvery evaluation pays for the dynamic dispatch, and Enzyme compiles a further " *
    "derivative for each call it differentiates at run time. Make the code these " *
    "lines name type stable. `ad_derivative_report` lists every finding."
end

function _derivative_message(::Val{:allocations}, derivative, findings, program)
    "ReactiveKernels: $(_derivative_description(derivative, program)) allocates:\n" *
    join(map(_derivative_finding_line, findings), "\n") *
    "\n`ad_allocations = :ignore`, the default, only lists a derivative's " *
    "allocations in `ad_derivative_report`; this preparation is checked with " *
    "`ad_allocations = :$(_ad_allocations())`."
end

_derivative_error(::Val{:unstable}, messages) = TypeInstabilityError(messages)
_derivative_error(::Val{:allocations}, messages) = DerivativeAllocationError(messages)

# Warn once per process about each derivative's findings of one kind
# (`:unstable` or `:allocations`), or refuse them.
_report_derivative_findings(kind, pairs, program, mode::Symbol) =
    _report_derivative_findings(kind, pairs, program, Val(mode))
_report_derivative_findings(kind, pairs, program, ::Val{:ignore}) = nothing
function _report_derivative_findings(kind, pairs, program, ::Val{:warn})
    for (derivative, findings) in _findings_by_derivative(pairs)
        id = (:derivative_finding, kind, derivative.func, derivative.mode,
              map(f -> (f.kind, f.callee, f.sweep, f.frames), findings))
        @warn _derivative_message(kind, derivative, findings, program) _group =
            :type_stability maxlog = 1 _id = id
    end
end
function _report_derivative_findings(kind, pairs, program, ::Val{:error})
    messages = [_derivative_message(kind, derivative, findings, program)
                for (derivative, findings) in _findings_by_derivative(pairs)]
    isempty(messages) || throw(_derivative_error(kind, messages))
end

# Check the report of one evaluation: its unstable calls with the
# `type_stability` setting `stability` (`:ignore` below `:all`), its
# allocations with `ad_allocations`. A report that holds no inspected
# derivative checked nothing, which is said once per process.
function _report_derivative(report::ADDerivativeReport, program::Type,
                            stability::Symbol, allocations::Symbol)
    if !any(d -> d.findings !== nothing, report.derivatives)
        notes = unique(String[d.note for d in report.derivatives])
        @warn "ReactiveKernels did not check the derivative native Enzyme compiles " *
              "for this preparation: " * (isempty(report.derivatives) ?
              "evaluating it once at the exemplar asked native Enzyme for no " *
              "derivative it reports." :
              "Enzyme did not inspect the derivatives it ran (" * join(notes, "; ") *
              ").") _group = :type_stability maxlog = 1 _id = (:derivative_uninspected, program)
        return nothing
    end
    _report_derivative_findings(Val(:unstable), report.unstable, program, stability)
    _report_derivative_findings(Val(:allocations), report.allocations, program, allocations)
    nothing
end
