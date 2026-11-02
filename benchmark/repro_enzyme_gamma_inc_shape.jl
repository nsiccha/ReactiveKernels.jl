# Backend-only reproducer: no RK code, derivative rule, or activity override.
# Run with Enzyme and SpecialFunctions in the active environment.
using Enzyme, SpecialFunctions, Test

lower(a, x) = log(first(gamma_inc(a, x)))
upper(a, x) = log(last(gamma_inc(a, x)))

println("Julia=",VERSION," Enzyme=",pkgversion(Enzyme),
    " SpecialFunctions=",pkgversion(SpecialFunctions))
println("Enzyme source=",pathof(Enzyme))

@testset "ordinary Reverse gamma_inc shape derivative" begin
    for f in (lower,upper), (a,x) in ((1.35,0.015),(1.35,0.35),(1.3,1.0))
        actual = Enzyme.autodiff(Reverse,f,Active,Active(a),Const(x))[1][1]
        finite_difference(h) = (f(a+h,x)-f(a-h,x))/(2h)
        reference = finite_difference(1e-5)
        finer = finite_difference(5e-6)
        println((;endpoint=nameof(f),a,x,actual,reference,finer))
        @test reference ≈ finer rtol=1e-7 atol=1e-9
        @test actual ≈ reference rtol=1e-6 atol=1e-8
    end
end
