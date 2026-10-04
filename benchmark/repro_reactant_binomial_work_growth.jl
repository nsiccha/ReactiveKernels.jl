# Capture the original indexed/broadcast Binomial models at varied data sizes.
# Run with the normal Reactant/Enzyme environment and an output directory argument.
# On managed hosts, use KB_COMPACT_KEEP_LOG=1 with kb-run-compact to retain logs.
# Inspect the complete executable modules with audit_reactant_executable_work.py.
# Numerical checks and structural observations are deliberately separate:
# any surviving while is insufficient, and equivalent batched array work is
# permitted by USER direction 1rvu25u. Exact inventory equality is diagnostic.
using Reactant, Test, Pkg, SHA, Libdl
Reactant.set_default_backend("cpu")
length(ARGS) == 1 || error("usage: julia repro_reactant_binomial_work_growth.jl OUTPUT_DIR")
const BINOMIAL_WORK_OUT = abspath(only(ARGS))
mkpath(BINOMIAL_WORK_OUT)

function _bw_definitions_only(ex)
    ex isa Expr && ex.head === :macrocall && ex.args[1] == Symbol("@testset") &&
        return :(nothing)
    return ex
end
Base.include(_bw_definitions_only, Main,
    joinpath(@__DIR__, "../packages/ReactiveKernelsPPL/test/test_plate_response_values.jl"))

println("JULIA ", VERSION, " CPU ", Sys.CPU_NAME, " THREADS ", Threads.nthreads())
Pkg.status(["ReactiveKernels", "ReactiveKernelsPPL", "Reactant", "Enzyme"])
for path in filter(p -> occursin(r"ReactantExtra|libEnzyme", p), Libdl.dllist())
    println("LIBRARY ", path, " SHA256 ", bytes2hex(open(sha256, path)))
end

function _bw_capture(fx)
    saved = deepcopy(fx.data)
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    kernel, ad = sampler.kernel, sampler.ad
    both(v) = ad_value_and_gradient(ad, v)
    ru = Reactant.to_rarray(fx.u)
    primal = Reactant.@compile sync=true kernel(ru)
    reverse = Reactant.@compile sync=true both(ru)
    n = length(fx.data[:y])
    for (name, executable) in (("primal", primal), ("reverse", reverse))
        text = repr(only(Reactant.XLA.get_hlo_modules(executable.exec)))
        path = joinpath(BINOMIAL_WORK_OUT, "$(fx.kind)-$(n)-$(name).hlo")
        write(path, text)
        println("CAPTURE ", basename(path), " bytes=", ncodeunits(text),
            " SHA256 ", bytes2hex(sha256(text)))
    end
    modules = (Reactant.@code_hlo(optimize=false, kernel(ru)),
        Reactant.@code_hlo(kernel(ru)),
        Reactant.@code_hlo(optimize=false, both(ru)),
        Reactant.@code_hlo(both(ru)))
    for (name, mod) in zip(("primal-raw", "primal-optimized",
            "reverse-raw", "reverse-optimized"), modules)
        write(joinpath(BINOMIAL_WORK_OUT, "$(fx.kind)-$(n)-$(name).mlir"), String(mod))
    end
    for shift in (-0.05, 0.0, 0.03)
        u = fx.u .+ shift
        r = Reactant.to_rarray(u)
        @test Float64(primal(r)) ≈ fx.oracle(u) rtol=1e-9
        value, gradient = reverse(r)
        @test Float64(value) ≈ fx.oracle(u) rtol=1e-9
        @test Array(gradient) ≈ _prv_fd(fx.oracle, u) rtol=1e-5 atol=1e-7
        @test Array(r) == u
    end
    @test fx.data == saved
    return nothing
end

@testset "original Binomial consumer: values, ordinary reverse and ownership" begin
    for indexed in (true, false)
        initial = _prv_bare(1; indexed)
        for n in (1, 2, 3, 9, 15, 33, 65)
            fx = n == 1 ? initial : _prv_bare(n; unbound=initial.unbound, indexed)
            Base.invokelatest(_bw_capture, fx)
        end
    end
end
println("COMPLETE: 28 executable HLO and 56 complete MLIR modules; " *
    "structural acceptance and timings are not claimed by the numerical testset.")
