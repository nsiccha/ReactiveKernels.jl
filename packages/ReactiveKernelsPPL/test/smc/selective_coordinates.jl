using ReactiveKernelsPPL, Test
import ReactiveKernelsPPL as PPL

function bounded_coordinates(n, width; definition=:(m+width))
    source=quote
        m ~ Normal(0,1)
        hi = $definition
        z[1:nb] .~ Uniform.(m,hi)
        y .~ Normal.(z[block],1)
    end
    data=(;nb=n,width,block=[1],y=[0.2])
    plan=bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
    PPL._smc_coordinate_records(build_kernel(plan).layout)
end

function event_coordinates(events, width, family, orientation)
    lhs = orientation === :rows ? :(eachrow(z[1:nb,1:width])) :
        :(eachcol(z[1:width,1:nb]))
    prior = family === :simplex ? :(Dirichlet(ones(width))) :
        :(Ordered(Normal(0,1),$width))
    source = quote
        $lhs .~ $prior
        y .~ Normal.(0,1)
    end
    data=(;nb=events,width,y=[0.2])
    bound=bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
    PPL._smc_coordinate_records(build_kernel(bound).layout)
end

@testset "Independent coupled events retain their shape across growth" begin
    for family in (:simplex,:ordered), orientation in (:rows,:cols)
        old=event_coordinates(2,3,family,orientation)
        grown=event_coordinates(3,3,family,orientation)
        matched=PPL._smc_match_coordinates(old,grown)
        @test sort(filter(!iszero,matched.old))==collect(eachindex(old))
        @test length(matched.fresh)==(family===:simplex ? 2 : 3)
        @test_throws ArgumentError PPL._smc_match_coordinates(old,
            event_coordinates(2,4,family,orientation))
    end
    function lkj_coordinates(events, width, uplo)
        source=quote
            @plate for k in levels(s)
                L[k] ~ LKJCholesky($width,2.0,$uplo)
            end
            y .~ Normal.(0,1)
        end
        data=(;s=collect(1:events),y=[0.2])
        bound=bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
        PPL._smc_coordinate_records(build_kernel(bound).layout)
    end
    for uplo in ('L','U')
        old=lkj_coordinates(2,3,uplo)
        grown=lkj_coordinates(3,3,uplo)
        matched=PPL._smc_match_coordinates(old,grown)
        @test sort(filter(!iszero,matched.old))==collect(eachindex(old))
        @test length(matched.fresh)==3
        @test_throws ArgumentError PPL._smc_match_coordinates(old,
            lkj_coordinates(2,4,uplo))
    end
end

@testset "Carried coordinates retain their transform meaning" begin
    old=bounded_coordinates(2,2.0)
    grown=bounded_coordinates(3,2.0)
    matched=PPL._smc_match_coordinates(old,grown)
    @test matched.old==[1,2,3,0]
    @test matched.fresh==[4]
    @test_throws ArgumentError PPL._smc_match_coordinates(old,bounded_coordinates(2,3.0))
    @test_throws ArgumentError PPL._smc_match_coordinates(old,
        bounded_coordinates(2,2.0;definition=:(2m+width)))
end

@testset "Independent external scalar plates may grow" begin
    function coordinates(n)
        sampling=(;rhs=:(ScalarLaw(0,1)),broadcast=true,constructor_broadcast=true)
        entry=PPL.LayoutEntry(:external_plate,nothing,:z,
            [Symbol(:z,".",i) for i in 1:n],1,n,:external,NaN,NaN,
            Int[],PPL.LayoutEntry[],sampling)
        PPL._smc_coordinate_records(PPL.LayoutTable([entry],n))
    end
    matched=PPL._smc_match_coordinates(coordinates(2),coordinates(3))
    @test matched.old==[1,2,0]
    @test matched.fresh==[3]
end
