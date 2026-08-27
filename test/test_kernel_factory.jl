# Inc3 factory/composition substrate tests. Isolated in a module so fixtures cannot
# shadow package exports in the shared Pkg.test Main.
module TestKernelFactory
using ReactiveKernels, Test
const RKS = ReactiveKernels

# Single-object owned/shared fixture: method `add!` writes `total`; `combined` derives
# from `total`; `scale` is a constant recipe; `seed` is an unwritten source.
@kernel accum(seed) = begin
    total = seed
    scale = 2.0
    combined = total * scale
    add!(x) = begin
        total += x
    end
end

@testset "Inc3 factory substrate" begin
    @testset "owned-vs-shared field derivation" begin
        os = RKS._kernel_factory_owned_shared(accum)
        # `total` is directly written; `combined` derives from it → both OWNED
        @test :total in os.owned
        @test :combined in os.owned
        # `scale` (constant recipe) and `seed` (unwritten source) are SHARED authority
        @test :scale in os.shared
        @test :seed in os.shared
        # exact partition of the field set
        @test isempty(intersect(os.owned, os.shared))
        @test union(os.owned, os.shared) == Set(RKS.kernel_port_names(accum))
        # the direct-write seed is exactly the mutated field
        @test RKS._kernel_factory_direct_writes(accum) == Set((:total,))
    end
end

end # module TestKernelFactory
