# Backend-only reproducer for Enzyme snag tuple-gradient-i-860b15ac.
# Run in an environment with Enzyme and DifferentiationInterface. On Julia
# 1.13.1, Enzyme dogfood f025967a, the out-of-place call succeeds; the mutating
# call throws MethodError because the no-context adapter calls gradient! with
# a tuple active argument. RK retains its out-of-place/copy path in this case.
using Enzyme, DifferentiationInterface

const DI = DifferentiationInterface
f(t) = sum(abs2, t[1]) + t[2][]^2
point = (ones(3), Ref(2.))
destination = (zeros(3), Ref(0.))
backend = AutoEnzyme(mode=Enzyme.Reverse)
preparation = DI.prepare_gradient(f, backend, point)
println((; julia=VERSION, enzyme=pkgversion(Enzyme), di=pkgversion(DI)))
println("out_of_place=", DI.value_and_gradient(f, preparation, backend, point))
println("in_place=", DI.value_and_gradient!(f, destination, preparation, backend, point))
