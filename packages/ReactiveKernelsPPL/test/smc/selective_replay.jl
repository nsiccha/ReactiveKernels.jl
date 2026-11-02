using Test
@testset "Fresh-process selective and growing-window continuation" begin
    mktempdir() do directory
        path=joinpath(directory,"selective.bin")
        fixture=joinpath(@__DIR__,"selective_process.jl")
        cmd=`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $fixture`
        @test success(pipeline(`$cmd write $path $ARGS`;stdout=stdout,stderr=stderr))
        @test success(pipeline(`$cmd read $path $ARGS`;stdout=stdout,stderr=stderr))
    end
end
