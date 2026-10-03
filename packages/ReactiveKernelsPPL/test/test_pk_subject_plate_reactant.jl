module PKSubjectPlateReactantTests
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Test

@testset "PK subject plate: explicit compiled matrix boundary" begin
    # Both axes grow. This valid capability check must become an unexpected
    # pass when fixed-size matrix batching is supported, prompting removal of
    # the documented limitation. Other errors fail the boundary assertions.
    for (G, N) in ((2, 3), (5, 7))
        s = build_linear_pk_schedule(repeat(1:G; inner=N),
            repeat(collect(1.0:N), G), repeat(1:G; inner=N),
            repeat(collect(0.0:N-1), G), fill(2.0, G*N))
        names = (:op_type, :op_dt, :op_amount, :op_interval, :op_count, :op_read_idx)
        cols = NamedTuple{names}(Tuple(getproperty(s, n) for n in names))
        k = prepare(ReactiveKernelsPPL._pk_conc_spec;
            bound=merge((ends=s.op_ends, log_F=zeros(length(s.op_type))), cols))
        q = log.([10., .1, .2, .3, .5])
        args = Tuple(Reactant.to_rarray(x; track_numbers=true) for x in q)
        result = try
            compiled = Reactant.@compile k(args...)
            Array(compiled(args...))
        catch err
            @test err isa MethodError
            description = sprint(showerror, err)
            @test occursin("similar", description) && occursin("SMatrix", description)
            nothing
        end
        @test_broken result !== nothing && isapprox(result, k(q...); rtol=1e-9)
    end
end
end
