# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode (static activity analysis, the default) fails
# when a NON-INLINED function returns a Float64 array read from constant
# data, bare or inside a tuple, named tuple or struct, and the caller meets
# it with an active value:
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
# The arguments are all `Const`, so the result is provably inactive.  The
# same function inlined differentiates fine, which is why the trigger looks
# arbitrary in practice: a guard whose error message interpolates a value
# (`throw(ArgumentError("need one schedule, got $(length(c))"))`) makes an
# unwrapping helper too large to inline, while the same guard with a
# literal message stays inlinable.  A non-inlined IDENTITY on a `Const`
# named tuple is worse: no error, and the reverse pass accumulates the
# adjoint into the constant data itself, so every later evaluation reads
# corrupted data.  `set_runtime_activity(Reverse)` differentiates every case
# below correctly.  This is the shape of a module function an `@rkppl`
# model calls with a bound schedule column (snag
# `interpolated-err-41772e14`).  Recorded on gordito, Enzyme 0.13.209,
# Julia 1.10.11, 2026-10-02.
using Enzyme

schedule() = [(; a = [1.0, 2.0], k = [1, 2])]

unwrap_interpolated(c) = length(c) == 1 ? only(c) :
    throw(ArgumentError("need one schedule, got $(length(c))"))
unwrap_literal(c) = length(c) == 1 ? only(c) :
    throw(ArgumentError("need one schedule"))
@noinline unwrap_noinline(c) = c[1]
@noinline identity_noinline(nt) = nt

reads(unwrap) = (c, b) -> (s = unwrap(c); sum(b .* s.a .+ s.k))

function run(label, f, data; mode = Reverse)
    snapshot = deepcopy(data)
    result = try
        "d/db = $(autodiff(mode, f, Active, Const(data), Active(0.1))[1][2])"
    catch err
        "$(nameof(typeof(err)))"
    end
    intact = data == snapshot ? "data intact" : "DATA MUTATED: $(data)"
    println(rpad(label, 40), rpad(result, 32), intact)
end

println("expected d/db = 3.0 for every case")
run("literal-message guard (inlined)", reads(unwrap_literal), schedule())
run("interpolated-message guard", reads(unwrap_interpolated), schedule())
run("@noinline c[1]", reads(unwrap_noinline), schedule())
run("@noinline identity on a named tuple",
    (nt, b) -> (s = identity_noinline(nt); sum(b .* s.a .+ s.k)),
    only(schedule()))
run("interpolated guard, runtime activity", reads(unwrap_interpolated),
    schedule(); mode = set_runtime_activity(Reverse))
