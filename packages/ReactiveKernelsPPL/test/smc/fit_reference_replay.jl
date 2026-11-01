using Test
@testset "Fresh-process fitted carry replay" begin
    mktempdir() do dir
        script = joinpath(@__DIR__,"fit_reference_process.jl")
        path = joinpath(dir,"checkpoint.bin")
        project = dirname(Base.active_project())
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $script write $path`)
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $script read $path`)
    end
end
