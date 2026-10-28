# Backend-only reproducer: no ReactiveKernels or PPL imports.
# Reactant 0.2.290 / Julia 1.10.12, CPU: default compilation fails with
# a one-element return where two are required. The raw trace is valid.
# Enzyme-JAX's sliceDotGeneralHelper floors (limit-start)/stride; [1:4:2]
# contains two elements, not one. Later rewrites can obscure this first error.
#
# julia --project=<Reactant environment> benchmark/repro_reactant_strided_dot_gather.jl
# ... only_enzyme  # diagnostic control; not default-compiler acceptance
using Reactant, Test

Reactant.set_default_backend("cpu")
const DESIGN = hcat(ones(5), collect(range(-0.4, 0.6; length=5)))
strided_dot_gather(u) = (DESIGN * u)[[2, 4]]

mode = isempty(ARGS) ? "default" : only(ARGS)
mode in ("default", "only_enzyme") || error("expected default or only_enzyme")
u = [0.2, -0.3]
ru = Reactant.to_rarray(u)
@test strided_dot_gather(u) ≈ [0.245, 0.095]
raw = repr(Reactant.@code_hlo optimize=false strided_dot_gather(ru))
if haskey(ENV, "RK_STRIDED_DOT_IR_DIR")
    mkpath(ENV["RK_STRIDED_DOT_IR_DIR"])
    write(joinpath(ENV["RK_STRIDED_DOT_IR_DIR"], "backend-raw.mlir"), raw)
end
println("STRIDED_DOT_BEGIN mode=", mode, " Julia=", VERSION,
        " Reactant=", pkgversion(Reactant))
flush(stdout)
compiled = Reactant.compile(strided_dot_gather, (ru,);
                           optimize=mode == "default" ? true : :only_enzyme)
@test Array(compiled(ru)) ≈ strided_dot_gather(u)
@test Array(ru) == u
println("STRIDED_DOT_PASS mode=", mode)
