using Test
@testset "Fresh-process adaptive tuning interruption and recovery" begin
    mktempdir() do directory
        script = joinpath(@__DIR__, "tuning_lifecycle_process.jl")
        project = dirname(Base.active_project())
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $script write $directory`)
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $script read $directory`)
    end
end
