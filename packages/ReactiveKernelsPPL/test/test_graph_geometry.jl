module GraphGeometryTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import DifferentiationInterface as DI
import Enzyme

@kernel identity_coordinates(u, shape, data) = begin
    return u
end
@kernel zero_jacobian(u, shape, data) = begin
    return 0.0
end
struct IdentityVector{D}
    data::D
end
ReactiveKernelsPPL.sampling_geometry(::Type{IdentityVector}, args, shape) =
    ParameterGeometry(shape, :(length($(args[1]))); support=:real,
        constrain=identity_coordinates, unconstrain=identity_coordinates,
        logjac=zero_jacobian)
ReactiveKernelsPPL.sampling_logdensity(::IdentityVector, w) = -sum(abs2, w)/2

@kernel exponential_coordinates(u, shape, data, parent; rate=1.0) = begin
    value = parent .+ exp.(u) ./ rate
    return value
end
@kernel exponential_inverse(value, shape, data, parent; rate=1.0) = begin
    u = log.(rate .* (value .- parent))
    return u
end
@kernel exponential_jacobian(u, shape, data, parent; rate=1.0) = begin
    value = sum(u) - length(u)*log(rate)
    return value
end
struct ShiftVector{D,P,R}
    data::D
    parent::P
    rate::R
end
ShiftVector(data, parent; rate=1.0) = ShiftVector(data, parent, rate)
ReactiveKernelsPPL.sampling_geometry(::Type{ShiftVector}, args, shape) =
    ParameterGeometry(shape, :(length($(args[end-1]))); support=:lower_bounded,
        constrain=exponential_coordinates, unconstrain=exponential_inverse,
        logjac=exponential_jacobian)
ReactiveKernelsPPL.sampling_logdensity(::ShiftVector, w) = -sum(abs2, w)/2

@kernel coupled_coordinates(u, shape, data) = begin
    shared::Float64 = sum(u)
    value = u .+ shared
    return value
end
@kernel coupled_inverse(value, shape, data) = begin
    value_sum::Float64 = sum(value)
    u = value .- value_sum/(length(value)+1)
    return u
end
@kernel coupled_jacobian(u, shape, data) = begin
    return log1p(length(u))
end
struct CoupledVector{D}
    data::D
end
ReactiveKernelsPPL.sampling_geometry(::Type{CoupledVector}, args, shape) =
    ParameterGeometry(shape, :(length($(args[1]))); support=:real,
        constrain=coupled_coordinates, unconstrain=coupled_inverse,
        logjac=coupled_jacobian)
ReactiveKernelsPPL.sampling_logdensity(::CoupledVector, w) = -sum(abs2, w)/2

function fixture(n; law=:identity)
    data = (;last=collect(1:n), key=[mod1(i, 2) for i in 1:n],
        y=[0.1i for i in 1:n], rate=1.7)
    rhs = law === :identity ? :(IdentityVector(last)) :
        law === :nonlinear ? :(ShiftVector(last, a; rate=rate)) : :(CoupledVector(last))
    ast = quote
        a ~ Normal(0, 1)
        w[1:length(last)] ~ $rhs
        y .~ Normal.(w, 1)
    end
    bound = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=(:y,)), data)
    built = build_kernel(bound)
    (; data, bound, built, u=vcat(0.15, [0.04i for i in 1:n]))
end

function oracle(fx, u, law)
    z = u[2:end]
    w = law === :identity ? z : law === :nonlinear ?
        u[1] .+ exp.(z)./fx.data.rate : z .+ sum(z)
    jacobian = law === :identity ? 0.0 : law === :nonlinear ?
        sum(z)-length(z)*log(fx.data.rate) : log1p(length(z))
    points = -log(2pi)/2 .- (fx.data.y .- w).^2 ./ 2
    prior = -log(2pi)/2-u[1]^2/2-sum(abs2,w)/2
    (;w, jacobian, points, posterior=prior+sum(points)+jacobian)
end
function finite_gradient(f, u)
    map(eachindex(u)) do i
        plus, minus = copy(u), copy(u)
        plus[i] += 1e-5; minus[i] -= 1e-5
        (f(plus)-f(minus))/2e-5
    end
end

@testset "graph geometry: strict rows, offsets, host and replay" begin
    for law in (:identity, :nonlinear, :coupled), n in (0, 3, 17)
        fx = fixture(n;law)
        saved = deepcopy((fx.data, fx.u))
        expected = oracle(fx, fx.u, law)
        @test coordinate_names(fx.built.layout) == [:a; [Symbol("w.$i") for i in 1:n]]
        values = constrain(fx.built.layout, fx.u)
        @test values.w ≈ expected.w
        @test unconstrain(fx.built.layout, values) ≈ fx.u
        @test logjac(fx.built.layout, fx.u) ≈ expected.jacobian
        source = kernel_expr(fx.bound, fx.built.layout)
        @test occursin("_sampling_graph_endpoint", string(source))
        @test !occursin("_sampling_constrain(", string(source))
        replayed = ReactiveKernelsPPL._eval_kernel_def(
            Meta.parse(string(Base.remove_linenums!(deepcopy(source)))))
        query = prepare_query((;spec=replayed, layout=fx.built.layout), fx.bound, :sampler)
        @test query(fx.u) ≈ expected.posterior
        whole = prepare_query(fx.built, fx.bound, :likelihood)
        @test whole(fx.u) ≈ sum(expected.points)
        if n > 0 && law === :coupled
            # The shared reduction reads the complete packed block. It must
            # not acquire the identity transform's selected-row contract.
            strict = try
                prepare_partition(fx.built, fx.bound, :y; by=:key, locality=:error)
            catch err
                # Current row analysis cannot recover the full shape of the
                # slice feeding this reduction. Admission may improve without
                # changing the actual dependence on the complete block.
                err isa ReactiveKernels.PartitionLocalityError || rethrow()
                nothing
            end
            if strict !== nothing
                @test partition_footprint(strict.kernel,fx.u,1,nothing).unconstrained == collect(2:n+1)
            end
            # Inspect the conservative footprint independently of admission.
            coupled = prepare_cell_query(fx.built, fx.bound, :y; locality=:ignore)
            @test partition_footprint(coupled.kernel,fx.u,1).unconstrained == collect(2:n+1)
        elseif n > 0
            partition = prepare_partition(fx.built, fx.bound, :y; by=:key, locality=:error)
            @test sum(partition(fx.u, k) for k in partition_labels(partition)) ≈ whole(fx.u)
            for label in partition_labels(partition)
                rows = findall(==(label), fx.data.key)
                @test partition(fx.u, label) ≈ sum(expected.points[rows])
                footprint = partition_footprint(partition.kernel, fx.u, label, nothing).unconstrained
                wanted = law === :identity ? rows .+ 1 :
                    law === :nonlinear ? [1; rows .+ 1] : collect(2:n+1)
                @test footprint == wanted
            end
        end
        @test (fx.data, fx.u) == saved
    end
end

@testset "ordinary Enzyme through composed parameter geometry" begin
    for law in (:identity, :nonlinear, :coupled)
        fx = fixture(3; law)
        backend = DI.AutoEnzyme(;mode=Enzyme.Reverse)
        sampler = prepare_sampler(fx.built, fx.bound, fx.u; backend)
        value, gradient = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
        @test value ≈ oracle(fx, fx.u, law).posterior
        @test gradient ≈ finite_gradient(u->oracle(fx,u,law).posterior, fx.u) atol=2e-8
        if law !== :coupled
            cell = prepare_cell_sampler(fx.built, fx.bound, :y, fx.u; backend, locality=:error)
            value, gradient = cell_value_and_gradient!(cell, similar(fx.u), fx.u, 1)
            @test value ≈ oracle(fx,fx.u,law).points[1]
            @test gradient ≈ finite_gradient(u->oracle(fx,u,law).points[1],fx.u) atol=2e-8
        end
    end
end

@kernel scalar_coordinates(u, shape, parent) = begin
    return parent + exp(u[1])
end
@kernel scalar_inverse(value, shape, parent) = begin
    return [log(value-parent)]
end
@kernel scalar_jacobian(u, shape, parent) = begin
    return u[1]
end
struct ScalarShift{P}
    parent::P
end
ReactiveKernelsPPL.sampling_geometry(::Type{ScalarShift}, args, shape) =
    ParameterGeometry((), 1; support=:lower_bounded, constrain=scalar_coordinates,
        unconstrain=scalar_inverse, logjac=scalar_jacobian)
ReactiveKernelsPPL.sampling_logdensity(::ScalarShift, w) = -w^2/2

struct JointIdentity{D}
    data::D
end
ReactiveKernelsPPL.sampling_geometry(::Type{JointIdentity}, args, shape) =
    ParameterGeometry((:(length($(args[1]))),), :(length($(args[1]))); support=:real,
        constrain=identity_coordinates, unconstrain=identity_coordinates, logjac=zero_jacobian)
ReactiveKernelsPPL.sampling_logdensity(::JointIdentity, w) = -sum(abs2,w)/2

struct WrongJoint{D}
    data::D
end
ReactiveKernelsPPL.sampling_geometry(::Type{WrongJoint},args,shape) =
    ParameterGeometry((:(length($(args[1]))),), :(length($(args[1]))); support=:real,
        constrain=zero_jacobian, unconstrain=identity_coordinates, logjac=zero_jacobian)
ReactiveKernelsPPL.sampling_logdensity(::WrongJoint,w) = -sum(abs2,w)/2

struct LocalGraphVector{D}
    data::D
end
function ReactiveKernelsPPL.sampling_geometry(::Type{LocalGraphVector}, args, shape)
    @kernel local_coordinates(u, shape, data) = begin
        return u
    end
    ParameterGeometry(shape, :(length($(args[1]))); support=:real,
        constrain=local_coordinates, unconstrain=local_coordinates, logjac=zero_jacobian)
end
ReactiveKernelsPPL.sampling_logdensity(::LocalGraphVector,w) = -sum(abs2,w)/2

struct OpaqueVector{D}
    data::D
end
ReactiveKernelsPPL.sampling_geometry(::Type{OpaqueVector},args,shape) =
    ParameterGeometry(shape, :(length($(args[1]))); support=:real,
        constrain=(u,shape,data)->u, unconstrain=(w,shape,data)->w,
        logjac=(u,shape,data)->0.0)
ReactiveKernelsPPL.sampling_logdensity(::OpaqueVector,w) = -sum(abs2,w)/2

@testset "local graph reconstruction and prepared default arguments" begin
    data = (;last=[1,2,3], y=[0.1,0.2,0.3], key=[1,1,2])
    for rhs in (:(LocalGraphVector(last)), :(ShiftVector(last,0.2)), :(OpaqueVector(last)))
        ast = quote
            w[1:length(last)] ~ $rhs
            y .~ Normal.(w,1)
        end
        bound = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(bound)
        u = [0.1,-0.2,0.3]
        nt = constrain(built.layout,u)
        @test unconstrain(built.layout,nt) ≈ u
        expected = -sum(abs2,nt.w)/2+logjac(built.layout,u)+
            sum(-log(2pi)/2 .- (data.y .- nt.w).^2 ./ 2)
        @test prepare_query(built,bound,:sampler)(u) ≈ expected
        if rhs.args[1] === :LocalGraphVector
            partition = prepare_partition(built,bound,:y;by=:key,locality=:error)
            @test partition_footprint(partition.kernel,u,1,nothing).unconstrained == [1,2]
        elseif rhs.args[1] === :OpaqueVector
            @test_throws ReactiveKernels.PartitionLocalityError prepare_partition(
                built,bound,:y;by=:key,locality=:error)
        end
    end
    small, large = fixture(3;law=:nonlinear), fixture(131;law=:nonlinear)
    @test length(small.built.spec.graph.recipes) == length(large.built.spec.graph.recipes)
end

struct SplatVector{D}
    data::D
end
ReactiveKernelsPPL.sampling_geometry(::Type{SplatVector},args,shape) =
    ParameterGeometry(shape, 3; support=:real, constrain=identity_coordinates,
        unconstrain=identity_coordinates, logjac=zero_jacobian)
ReactiveKernelsPPL.sampling_logdensity(::SplatVector,w) = -sum(abs2,w)/2
geometry_keywords(rate) = (;rate)

@testset "splatted calls preserve numerical endpoint lowering" begin
    data = (;last=[1,2,3],y=[0.1,0.2,0.3],rate=1.7)
    u = [0.1,-0.2,0.3]
    for rhs in (:(SplatVector((last,)...)), :(ShiftVector(last,0.2;geometry_keywords(rate)...)))
        bound = bind_data(lower_rkppl(quote
            w[1:3] ~ $rhs
            y .~ Normal.(w,1)
        end,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(bound)
        nonlinear = rhs.args[1] === :ShiftVector
        w = nonlinear ? 0.2 .+ exp.(u)./data.rate : u
        jac = nonlinear ? sum(u)-length(u)*log(data.rate) : 0.0
        expected = -sum(abs2,w)/2+jac+sum(-log(2pi)/2 .- (data.y .- w).^2 ./ 2)
        source = kernel_expr(bound,built.layout)
        @test !occursin("_sampling_graph_endpoint",string(source))
        @test prepare_query(built,bound,:sampler)(u) ≈ expected
        replay = ReactiveKernelsPPL._eval_kernel_def(
            Meta.parse(string(Base.remove_linenums!(source))))
        @test prepare_query((;spec=replay,layout=built.layout),bound,:sampler)(u) ≈ expected
        if nonlinear
            nt = constrain(built.layout,u)
            @test nt.w ≈ w
            @test unconstrain(built.layout,nt) ≈ u
            @test logjac(built.layout,u) ≈ jac
        end
    end
end

@testset "graph endpoints in scalar and joint cells" begin
    for n in (0,3)
        data = (;parents=fill(0.2,n), y=fill(0.4,n))
        ast = quote
            @plate for i in eachindex(parents)
                w[i] ~ ScalarShift(parents[i])
                y[i] ~ Normal(w[i],1)
            end
        end
        bound = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(bound)
        u = fill(-0.3,n)
        nt = constrain(built.layout,u)
        @test nt.w ≈ data.parents .+ exp.(u)
        @test unconstrain(built.layout,nt) ≈ u
        @test logjac(built.layout,u) ≈ sum(u)
        expected = -sum(abs2,nt.w)/2 + sum(u) + sum(-log(2pi)/2 .- (data.y .- nt.w).^2 ./ 2)
        @test prepare_query(built,bound,:sampler)(u) ≈ expected
        if n > 0
            sampler = prepare_sampler(built,bound,u;backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
            value, gradient = sampler_value_and_gradient!(sampler,similar(u),u)
            @test value ≈ expected
            @test gradient ≈ (data.y .- 2nt.w).*exp.(u) .+ 1
        end
    end
    for last in ([Float64[],[0.0,0.0],[0.0]], [[0.0],[0.0,0.0,0.0]])
        data = (;last, y=[fill(0.4,length(v)) for v in last])
        saved = deepcopy(data)
        ast = quote
            @plate for i in eachindex(last)
                w[i] ~ JointIdentity(last[i])
                y[i] .~ Normal.(w[i],1)
            end
        end
        bound = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(bound)
        u = [0.05i for i in 1:built.layout.total]
        nt = constrain(built.layout,u)
        @test length.(nt.w) == length.(last)
        @test reduce(vcat,nt.w) ≈ u
        @test unconstrain(built.layout,nt) ≈ u
        @test logjac(built.layout,u) == 0
        source = kernel_expr(bound,built.layout)
        @test occursin("_sampling_graph_endpoint",string(source))
        expected = -sum(abs2,u)/2 + sum(-log(2pi)/2 .- (0.4 .- u).^2 ./ 2)
        @test prepare_query(built,bound,:sampler)(u) ≈ expected
        sampler = prepare_sampler(built,bound,u;backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler,similar(u),u)
        @test value ≈ expected
        @test gradient ≈ 0.4 .- 2u
        @test data == saved
    end
    data = (;last=[[0.0,0.0]])
    bound = bind_data(lower_rkppl(quote
        @plate for i in eachindex(last)
            w[i] ~ WrongJoint(last[i])
        end
    end,data;mod=@__MODULE__),data)
    built = build_kernel(bound)
    @test_throws "its shape there is (2,)" prepare_query(built,bound,:sampler)(zeros(2))
end

end
