module ScanEndpointScopeReactantTests
using ReactiveKernels, Reactant, DifferentiationInterface, Enzyme, Test
include(joinpath(@__DIR__, "fixtures", "scan_endpoint_scope.jl"))
const F = ScanEndpointScopeFixtures
Reactant.set_default_backend("cpu")

function inventory(text, pattern)
    names = [m.captures[1] for m in eachmatch(pattern, text)]
    sort!(collect(Dict(name => count(==(name), names) for name in unique(names)));
          by=first)
end

function record_structure(fn, compiled, args, label, n)
    mlir = repr(Reactant.@code_hlo fn(args...))
    hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
    raw = repr(Reactant.@code_hlo optimize=false fn(args...))
    # Enforce RK emission, including bound-data traces. Backend optimization
    # may unroll the loop while preserving the tested values and derivatives.
    @test occursin("stablehlo.while", raw)
    startswith(label, "primal") && @test count("stablehlo.while", raw) == 1
    println("SCAN_ENDPOINT_RAW ", label, " n=", n,
            " while=", count("stablehlo.while", raw))
    println("SCAN_ENDPOINT_MLIR ", label, " n=", n, " ", inventory(mlir,
        r"\b((?:stablehlo|chlo|enzyme|func|arith|scf|cf|tensor|math|linalg|memref)\.\w+)"))
    instructions = r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\("
    @test count(instructions, hlo) == count(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = ", hlo)
    println("SCAN_ENDPOINT_HLO ", label, " n=", n, " ", inventory(hlo, instructions))
    if haskey(ENV, "RK_SCAN_ENDPOINT_IR_DIR")
        stem = joinpath(ENV["RK_SCAN_ENDPOINT_IR_DIR"], "$label-$n")
        write(stem * ".raw.mlir", raw)
        write(stem * ".mlir", mlir)
        write(stem * ".hlo", hlo)
    end
end

@testset "object endpoint scans emit loops and preserve compiled primal and reverse" begin
    q = [0.7]
    rq = Reactant.to_rarray(q)
    for n in (3, 17, 31), bound in (false, true)
        xs = sin.(1:n)
        saved_q, saved_xs = copy(q), copy(xs)
        k = bound ? prepare(F.scaled_scan; bound=(; xs)) : prepare(F.scaled_scan)
        args = bound ? (q,) : (q, xs)
        traced = bound ? (rq,) : (rq, Reactant.to_rarray(xs))
        weight = sum((n - i + 1) * xs[i] for i in eachindex(xs))
        primal = Reactant.@compile k(traced...)
        @test Float64(primal(traced...)) ≈ q[1] * weight
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), args...; active=:q)
        gradient_fn = bound ?
            (p -> only(Enzyme.gradient(Enzyme.Reverse, k, p))) :
            ((p, x) -> only(Enzyme.gradient(Enzyme.Reverse, t -> k(t, x), p)))
        reverse = Reactant.@compile gradient_fn(traced...)
        @test Array(reverse(traced...)) ≈ [weight]
        @test Array(reverse(traced...)) ≈ ad_gradient(ad, args...)
        record_structure(k, primal, traced, "primal-$bound", n)
        record_structure(gradient_fn, reverse, traced, "reverse-$bound", n)
        @test q == saved_q && xs == saved_xs
        @test Array(rq) == q
        bound || @test Array(traced[2]) == xs
    end
    # Establish that the small-loop expansion predates endpoint scope binding.
    for spec in (F.scaled_scan, F.scaled_function_scan)
        xs = sin.(1:3)
        k = prepare(spec)
        traced = (rq, Reactant.to_rarray(xs))
        primal = Reactant.@compile k(traced...)
        @test Float64(primal(traced...)) ≈ k(q, xs)
        raw = repr(Reactant.@code_hlo optimize=false k(traced...))
        @test count("stablehlo.while", raw) == 1
    end
end

@testset "computed and lazy endpoint arguments keep scan semantics" begin
    for spec in (F.object_scan, F.computed_scan, F.shifted_scan, F.lazy_scan)
        xs = [-1.0, 0.25, 0.0, 4.0, -2.0]
        k = prepare(spec)
        rx = Reactant.to_rarray(xs)
        compiled = Reactant.@compile k(rx)
        @test Array(compiled(rx)) ≈ k(xs)
        @test Array(rx) == xs
    end
    xs = [-1.0, 0.25, 0.0, 4.0, -2.0]
    k = prepare(F.formal_argument_scan)
    rx = Reactant.to_rarray(xs)
    compiled = Reactant.@compile k(rx, 0.75)
    @test Array(compiled(rx, 0.75)) ≈ k(xs, 0.75)
    @test Array(rx) == xs
    k = prepare(F.object_scan; want=:total)
    rx = Reactant.to_rarray(Float64[])
    compiled = Reactant.@compile k(rx)
    @test Float64(compiled(rx)) === 0.0
end
end
