using Test
@testset "Fresh-process moment-calibrated window replay" begin
    mktempdir() do dir
        file=joinpath(dir,"checkpoint.bin")
        script=joinpath(@__DIR__,"moment_reference_process.jl")
        project=dirname(Base.active_project())
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $script write $file`)
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $script read $file`)
    end
end
