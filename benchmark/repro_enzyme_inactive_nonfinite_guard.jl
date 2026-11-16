# Backend-only reproduction: no ReactiveKernels lowering or custom AD rules.
# Run with: julia --project=<environment containing Enzyme> <this file>
# On Enzyme 9430736f / libEnzyme eb82b697 / Julia 1.10.12, the rejected
# nonfinite inputs return -Inf but ordinary Reverse produces NaN, not zero.
using Enzyme

function guarded_integer(x::Float64, n::Int)
    if isfinite(n) && isinteger(n) && n >= 0 && isfinite(x) && 0 <= x && x <= 1
        x * x
    else
        -Inf
    end
end

println("Enzyme source: ", pkgdir(Enzyme))
println("Enzyme version: ", pkgversion(Enzyme))
println("libEnzyme: ", Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
for x in (0.4, -0.2, 1.2, NaN, Inf, -Inf)
    gradient = Enzyme.autodiff(Enzyme.Reverse, guarded_integer, Enzyme.Active,
        Enzyme.Active(x), Enzyme.Const(5))[1][1]
    println((x=x, value=guarded_integer(x, 5), gradient=gradient))
end
