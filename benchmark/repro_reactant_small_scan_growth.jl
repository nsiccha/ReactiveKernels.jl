# Backend-only reproduction: RK emits a retained scan, but the default
# optimizer expands short recurrences without violating RK emission requirements.
# No ReactiveKernels dependency, activity
# override, derivative rule, or nondefault executable pipeline is used.
# Run in an environment containing Reactant, Enzyme and Test:
#   julia --project=<env> benchmark/repro_reactant_small_scan_growth.jl
# Set RK_SMALL_SCAN_IR_DIR to preserve complete raw/default MLIR and actual HLO.
# Optional --isolate-unroll tests the single unroll rewrite after HLO
# simplification, using the documented custom optimize string for MLIR only.
# This diagnostic does not select a nondefault executable pipeline.
module SmallScanGrowthReproducer
using Reactant, Enzyme, Test
Reactant.set_default_backend("cpu")

function scan_loss(u)
    n = length(u) - 2
    phi = Reactant.@allowscalar u[2]
    z = u[3:end]
    buffer = zero.(z)
    carry = Reactant.@allowscalar z[1]
    Reactant.@allowscalar buffer[1] = carry
    Reactant.@trace for i in 2:n
        zi = Reactant.@allowscalar z[i]
        carry = phi * carry + zi
        Reactant.@allowscalar buffer[i] = carry
    end
    -sum(abs2, u) / 2 + sum(buffer) - sum(exp.(buffer))
end

# Independent native recurrence; ordinary Enzyme differentiates this oracle.
function oracle(u)
    h = accumulate((h, e) -> u[2] * h + e, u[3:end])
    -sum(abs2, u) / 2 + sum(h) - sum(exp.(h))
end
reverse_loss(u) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(scan_loss), u))

const MLIR_OP = r"\b((?:stablehlo|chlo|enzyme|enzymexla|func|arith|scf|cf|tensor|math|linalg|memref)\.[\w]+)"
const HLO_OP = r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\("
const HLO_INSTRUCTION = r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = "
# MLIR's custom text also prints dialect attributes and nested region ops;
# report the complete dialect-token inventory separately from HLO instructions.
function inventory(source, pattern)
    counts = Dict{String,Int}()
    for m in eachmatch(pattern, source)
        name = m.captures[1]
        counts[name] = get(counts, name, 0) + 1
    end
    sort!(collect(counts); by=first)
end
function preserve(name, source)
    if haskey(ENV, "RK_SMALL_SCAN_IR_DIR")
        write(joinpath(ENV["RK_SMALL_SCAN_IR_DIR"], name), source)
    end
end

function check_result(::Val{:primal}, compiled, ru, u)
    @test Float64(compiled(ru)) ≈ oracle(u) rtol=1e-12 atol=1e-12
end
function check_result(::Val{:reverse}, compiled, ru, u)
    expected = only(Enzyme.gradient(Enzyme.Reverse, oracle, u))
    @test Array(compiled(ru)) ≈ expected rtol=1e-12 atol=1e-12
end

function main()
@testset "default short-scan semantics and optimization diagnostics" begin
    println("VERSIONS Julia=", VERSION, " Reactant=", pkgversion(Reactant),
            " Enzyme=", pkgversion(Enzyme))
    for n in (3, 4, 8, 16)
        u = [0.2, -0.3, [0.1sin(i) for i in 1:n]...]
        ru = Reactant.to_rarray(u)
        for (direction, fn) in ((:primal, scan_loss), (:reverse, reverse_loss))
            raw = repr(Reactant.@code_hlo optimize=false fn(ru))
            mlir = repr(Reactant.@code_hlo fn(ru))
            compiled = Reactant.@compile fn(ru)
            hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
            @test count(HLO_OP, hlo) == count(HLO_INSTRUCTION, hlo)
            @test occursin("stablehlo.while", raw)
            for point in (u, [-0.15, 0.4, [0.07cos(i) for i in 1:n]...])
                saved = copy(point)
                rp = Reactant.to_rarray(point)
                check_result(Val(direction), compiled, rp, point)
                @test point == saved
                @test Array(rp) == saved
            end
            # RK's contract concerns emitted structure; final backend loop
            # retention is diagnostic, not a broken capability expectation.
            println("SCAN n=", n, " direction=", direction,
                    " raw_while=", count("stablehlo.while", raw),
                    " default_while=", count("stablehlo.while", mlir),
                    " executable_while=", count(r"\bwhile\(", hlo))
            println("MLIR_DIALECT_TOKEN_INVENTORY ", inventory(mlir, MLIR_OP))
            println("HLO_INVENTORY ", inventory(hlo, HLO_OP))
            stem = "scan-n$(n)-$(direction)"
            preserve(stem * "-raw.mlir", raw)
            preserve(stem * "-default.mlir", mlir)
            preserve(stem * ".hlo", hlo)
            flush(stdout)
        end
    end
end
end

function diagnose_unroll()
    @testset "isolated short-loop rewrite diagnostic" begin
        control = "inline,canonicalize,enzyme-hlo-opt"
        rewrite = control * ",enzyme-hlo-generate-td{patterns=enzyme_hlo_unroll(4);}," *
                  "transform-interpreter,enzyme-hlo-remove-transform,canonicalize"
        for n in (3, 8)
            ru = Reactant.to_rarray([0.2, -0.3, [0.1sin(i) for i in 1:n]...])
            before = repr(Reactant.@code_hlo optimize=control scan_loss(ru))
            after = repr(Reactant.@code_hlo optimize=rewrite scan_loss(ru))
            @test count("stablehlo.while", before) == 1
            @test count("stablehlo.while", after) == (n == 3 ? 0 : 1)
            println("ISOLATED_UNROLL n=", n, " before_while=",
                    count("stablehlo.while", before), " after_while=",
                    count("stablehlo.while", after))
            preserve("isolate-n$(n)-control.mlir", before)
            preserve("isolate-n$(n)-unroll.mlir", after)
        end
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
    "--isolate-unroll" in ARGS && diagnose_unroll()
end
end
