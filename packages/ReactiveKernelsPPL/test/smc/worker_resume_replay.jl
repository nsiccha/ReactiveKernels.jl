using Test
@testset "Tuning and production resume across process worker counts" begin
    script = joinpath(@__DIR__,"worker_resume_process.jl")
    project = dirname(Base.active_project())
    workers = min(4,Threads.nthreads())
    for (writer,reader) in unique(((1,workers),(workers,1)))
        mktempdir() do directory
            command = `$(Base.julia_cmd()) --startup-file=no --project=$project`
            @test success(`$command --threads=$writer $script write $directory`)
            @test success(`$command --threads=$reader $script read $directory`)
        end
    end
end
