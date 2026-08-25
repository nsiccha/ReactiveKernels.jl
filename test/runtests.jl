using ReactiveKernels
using Test

# Minimal smoke suite — establishes the test harness (test/Project.toml + this
# file are wired into `test.yml`). The implementer expands this as the
# have/want planning API lands.
@testset "ReactiveKernels.jl" begin
    @test ReactiveKernels isa Module
end
