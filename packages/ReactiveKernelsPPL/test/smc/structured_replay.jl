using Test
@testset "Fresh-process structured reference and fixed metric continuation" begin
    mktempdir() do directory
        path = joinpath(directory,"structured.bin")
        fixture = joinpath(@__DIR__,"structured_process.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(dirname(Base.active_project())) $fixture`
        @test success(pipeline(`$cmd write $path`;stdout=stdout,stderr=stderr))
        @test success(pipeline(`$cmd read $path`;stdout=stdout,stderr=stderr))
    end
end
