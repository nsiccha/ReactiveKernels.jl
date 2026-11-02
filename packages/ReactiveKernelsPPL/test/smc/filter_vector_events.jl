@testset "Vector events preserve broadcasts and group input axes" begin
    model = @rkppl begin
        offset ~ Normal()
        @plate for g in eachindex(y)
            loc[g] = offset .+ x[g]
            y[g] ~ MvNormal(loc[g], covariance)
        end
    end
    x = [[1.0, 2.0], [3.0, 4.0]]
    y = [[1.2, 2.3], [3.4, 4.5]]
    covariance = [1.0 0.2; 0.2 0.7]
    expected = sum(logpdf(MvNormal(0.2 .+ a, covariance), b) for (a,b) in zip(x,y))
    fixed = prepare_filter(model; model_id="fixed-vector-events", parameters=(;offset=0.2),
        data=(;x,covariance), groups=(;g=1:2))
    state = initialize_filter(fixed; rng=Xoshiro(1), particles=2)
    @test filter_step(state, (;y)).predictive_logscore ≈ expected
    bound = model(;x,covariance,offset=0.2) | (;y)
    @test Base.invokelatest(prepare_query(build_kernel(bound),bound,:likelihood),Float64[]) ≈ expected

    varying = prepare_filter(model; model_id="varying-vector-events", parameters=(;offset=0.2),
        data=(;covariance), inputs=(:x,), groups=(;g=1:2))
    state = initialize_filter(varying; rng=Xoshiro(2), particles=2)
    @test filter_step(state, (;y); inputs=(;x)).predictive_logscore ≈ expected
    after = filter_step(state, (;y); inputs=(;x), groups=(;g=[2]))
    @test after.predictive_logscore ≈ logpdf(MvNormal(0.2 .+ x[2],covariance),y[2])
    @test after.group_times.g == [0,1]

    independent = @rkppl begin
        @plate for g in eachindex(y)
            y[g] .~ Normal.(x[g] .+ 0.2, 0.5)
        end
    end
    target = prepare_filter(independent; model_id="broadcast-vector-events", data=(;x), groups=(;g=1:2))
    state = initialize_filter(target; rng=Xoshiro(3), particles=2)
    @test filter_step(state,(;y)).predictive_logscore ≈
        sum(sum(logpdf.(Normal.(a .+ 0.2,0.5),b)) for (a,b) in zip(x,y))
    checkpoint = filter_checkpoint(state)
    @test_throws ArgumentError filter_step(state,(;y=[[0.1],y[2]]))
    @test filter_checkpoint(state) == checkpoint
end

const filter_event_density = @kernel function filter_event_density(y, loc)
    return -0.5 * (y-loc)^2 - 0.5log(2pi)
end

@testset "Broadcast kernel densities and nested laws" begin
    model = @rkppl begin
        y .~ LogDensity.(filter_event_density,x)
    end
    target = prepare_filter(model;model_id="broadcast-kernel-density",inputs=(:x,))
    state = initialize_filter(target;rng=Xoshiro(3),particles=2)
    @test filter_step(state,(;y=0.4);inputs=(;x=0.2)).predictive_logscore ≈ logpdf(Normal(0.2),0.4)
    @test filter_step(state,(;y=[0.4]);inputs=(;x=0.2)).predictive_logscore ≈ logpdf(Normal(0.2),0.4)
    x, y = [0.2,-0.1], [0.4,0.3]
    @test filter_step(state,(;y);inputs=(;x)).predictive_logscore ≈ sum(logpdf.(Normal.(x),y))
    model = @rkppl begin
        y ~ truncated(Normal(x,1.0),-1.0,1.0)
    end
    target = prepare_filter(model;model_id="nested-current-law",inputs=(:x,))
    state = initialize_filter(target;rng=Xoshiro(3),particles=2)
    @test filter_step(state,(;y=0.4);inputs=(;x=0.2)).predictive_logscore ≈
        logpdf(truncated(Normal(0.2),-1.0,1.0),0.4)
end
