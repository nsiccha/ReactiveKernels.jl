module TestJSRun

using ReactiveKernels
using Test

const NODE = ReactiveKernels._js_default_node() !== nothing
const js_run = ReactiveKernels.js_run

@kernel js_kinds(x::Vector{Float64}, k::Int, flag::Bool) = begin
    n = length(x)
    y = x .* k
    z = !flag
    return y
end

# Hand-written modules for the runner's own contract, beside the emitted ones.
const ECHO = """
export class JuliaError extends Error {
  constructor(type, message) { super(message); this.name = type; }
}
export const INPUTS = [{ name: "x", type: "Vector{Float64}" }];
export const OUTPUTS = [{ name: "x", type: "Vector{Float64}" }];
export const RETURNS = "x";
export function echo(x) { return Array.from(x); }
export function kind(x) { return ArrayBuffer.isView(x) ? [1] : [0]; }
export function fail(x) { throw new JuliaError("DomainError", "naïve \\"quoted\\" — 𝑥 = " + x[0]); }
export function bug(x) { return x.nothing.here; }
export function mutate(x) { x[0] = 42; return Array.from(x); }
export function spin(x) { for (;;) {} }
"""

@testset "js_run" begin
    if !NODE
        @test_skip "no node on the PATH"
    else
        @testset "values come back in the native shape and types" begin
            want = (count = :n, scaled = :y, negated = :z)
            source = ReactiveKernels.js_source(js_kinds; have = (:x, :k, :flag), want,
                                               types = (; x = Vector{Float64}, k = Int, flag = Bool),
                                               name = :kinds)
            native = prepare(js_kinds; have = (:x, :k, :flag), want = (:n, :y, :z))
            cases = [(Float64[], 2, true), ([1.5], -3, false), (collect(range(0, 1; length = 1000)), 7, true)]
            for vectors in (:array, :float64array)
                outcomes = js_run(source, :kinds, cases; vectors)
                @test length(outcomes) == length(cases)
                for (c, o) in zip(cases, outcomes)
                    n, y, z = native(c...)
                    @test o.error === nothing && !o.inputs_mutated
                    @test o.value isa NamedTuple{(:count, :scaled, :negated),Tuple{Int64,Vector{Float64},Bool}}
                    @test o.value.count === n && o.value.scaled == y && o.value.negated === z
                end
            end
            # A tuple WANT gives a tuple; a name gives the value.
            tuple = ReactiveKernels.js_source(js_kinds; have = (:x, :k, :flag), want = (:n, :z),
                                              types = (; x = Vector{Float64}, k = Int, flag = Bool))
            @test only(js_run(tuple, :kernel, [([1.0, 2.0], 1, false)])).value === (2, true)
        end

        @testset "every WANT form comes back as native prepare returns it" begin
            have = (:x, :k, :flag)
            types = (; x = Vector{Float64}, k = Int, flag = Bool)
            cases = [(Float64[], 2, true), ([1.5, -2.0], -3, false)]
            for want in (:n, (:n,), (:n, :z), (count = :n, scaled = :y, negated = :z),
                         (a = (:n, :z), b = (c = :y,)), (:n, (s = :y,)), (first = :n, again = :n),
                         (one = :z,), (:y, :n))
                source = ReactiveKernels.js_source(js_kinds; have, want, types)
                native = prepare(js_kinds; have, want)
                for (c, o) in zip(cases, js_run(source, :kernel, cases))
                    expected = native(c...)
                    @test o.error === nothing
                    @test typeof(o.value) == typeof(expected) && isequal(o.value, expected)
                end
            end
        end

        @testset "numbers cross exactly, nonfinite ones and -0.0 included" begin
            x = [NaN, Inf, -Inf, -0.0, 0.1, 1e-310, prevfloat(1.0), 2.0^53]
            o = only(js_run(ECHO, :echo, [(x,)]))
            @test isequal(o.value, x) && signbit(o.value[4])
            @test only(js_run(ECHO, :kind, [(x,)]; vectors = :array)).value == [0.0]
            @test only(js_run(ECHO, :kind, [(x,)]; vectors = :float64array)).value == [1.0]
        end

        @testset "errors thrown by a call are its outcome" begin
            o = only(js_run(ECHO, :fail, [([1.5],)]))
            @test o.value === nothing && o.julia_error
            @test o.error == "DomainError" && o.message == "naïve \"quoted\" — 𝑥 = 1.5"
            o = only(js_run(ECHO, :bug, [([1.0],)]))
            @test o.error == "TypeError" && !o.julia_error && !isempty(o.message)
        end

        @testset "the caller's arrays are never changed; a module's changes are reported" begin
            x = [1.0, 2.0]
            for vectors in (:array, :float64array)
                o = only(js_run(ECHO, :mutate, [(x,)]; vectors))
                @test o.inputs_mutated && o.value == [42.0, 2.0]
            end
            @test x == [1.0, 2.0]
            @test !only(js_run(ECHO, :echo, [(x,)])).inputs_mutated
        end

        @testset "a failed run throws JSRunError" begin
            failure(f) = try
                f()
                nothing
            catch e
                e
            end
            e = failure(() -> js_run(ECHO, :missing_entry, [([1.0],)]))
            @test e isa ReactiveKernels.JSRunError && occursin("no function `missing_entry`", e.message)
            e = failure(() -> js_run("export function f( {", :f, [()]))
            @test e isa ReactiveKernels.JSRunError && occursin("does not import", e.message)
            e = failure(() -> js_run("process.exit(3);\nexport function f() { return 1; }", :f, [()]))
            @test e isa ReactiveKernels.JSRunError && occursin("code 3", e.message)
            e = failure(() -> js_run(ECHO, :spin, [([1.0],)]; timeout = 2))
            @test e isa ReactiveKernels.JSRunError && occursin("no answer within", e.message)
            e = failure(() -> js_run(ECHO, :echo, [([1.0],)]; node = joinpath(tempdir(), "no-such-node")))
            @test e isa ReactiveKernels.JSRunError && occursin("could not be started", e.message)
            e = failure(() -> js_run(ECHO, :echo, [([1.0],)]; node = nothing))
            @test e isa ReactiveKernels.JSRunError && occursin("not available", e.message)
        end

        @testset "arguments without a JavaScript form are refused" begin
            @test_throws ArgumentError js_run(ECHO, :echo, [(["a" => 1],)])
            @test_throws ArgumentError js_run(ECHO, :echo, [(2^60,)])
            @test_throws ArgumentError js_run(ECHO, :echo, [([1.0],)]; vectors = :typed)
        end
    end
end

end
