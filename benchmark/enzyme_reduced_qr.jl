# Julia 1.10.12 / Enzyme 0.13.221 (b80617b) raises
# IllegalTypeAnalysisException in LinearAlgebra.getproperty(::QRCompactWY, ::Symbol).
# Run in an environment containing Enzyme: julia --project benchmark/enzyme_reduced_qr.jl
using Enzyme, LinearAlgebra, Random

function signed_factor_sum(A)
    F = qr(A)
    Q, R = Matrix(F.Q), F.R
    signs = sign.(diag(R))
    sum(Q .* transpose(signs)) + sum(signs .* R)
end

A = randn(MersenneTwister(430), 6, 3)
println((julia=VERSION, enzyme=pkgversion(Enzyme), value=signed_factor_sum(A)))
display(only(Enzyme.gradient(Enzyme.Reverse, signed_factor_sum, A)))
