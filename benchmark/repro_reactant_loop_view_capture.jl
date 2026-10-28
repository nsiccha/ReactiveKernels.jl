# Backend-only diagnostic: a read-only SubArray captured by @trace fails in
# recursive tracing of its offset1 field. An owned-array control preserves
# values and inputs. No ReactiveKernels code or derivative rules are loaded.
using Reactant, Test
Reactant.set_default_backend("cpu")

function raw_view_loop(x)
    w = view(x, 1:3)
    s = copy(x)
    Reactant.@trace for i in 1:3
        s = s .+ sum(w)
    end
    s
end

function raw_owned_loop(x)
    w = copy(view(x, 1:3))
    s = copy(x)
    Reactant.@trace for i in 1:3
        s = s .+ sum(w)
    end
    s
end

x = [0.2, -0.1, 0.3, 0.9]
saved = copy(x)
rx = Reactant.to_rarray(x)
@test raw_view_loop(x) == raw_owned_loop(x)
try
    exe = Reactant.compile(raw_view_loop, (rx,))
    @test Array(exe(rx)) ≈ raw_view_loop(x) rtol=2e-13 atol=2e-13
    println("RAW_VIEW_LOOP_SUPPORTED")
catch err
    message = sprint(showerror, err)
    err isa AssertionError && occursin("offset1", message) || rethrow()
    println("RAW_VIEW_LOOP_RECURSIVE_TRACER_FAILURE: ", message)
end
exe = Reactant.compile(raw_owned_loop, (rx,))
@test Array(exe(rx)) ≈ raw_owned_loop(x) rtol=2e-13 atol=2e-13
@test isequal(x, saved)
@test isequal(Array(rx), saved)
println("OWNED_CONTROL_PARITY_AND_INPUTS_PASS")
