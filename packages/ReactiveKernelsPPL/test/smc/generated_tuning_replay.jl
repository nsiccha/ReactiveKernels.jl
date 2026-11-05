using Test
@testset "Generated-module SMC checkpoint recovery" begin
    mktempdir() do directory
        script = joinpath(@__DIR__, "generated_tuning_process.jl")
        project = dirname(Base.active_project())
        command = `$(Base.julia_cmd()) --startup-file=no --project=$project $script`
        @test success(pipeline(`$command write $directory`; stdout=stdout, stderr=stderr))
        @test success(pipeline(`$command read $directory`; stdout=stdout, stderr=stderr))
    end
end
