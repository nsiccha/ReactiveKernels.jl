module GraphGeometryConstructorTests
using ReactiveKernels, ReactiveKernelsPPL, Test

struct TypedVector{T,D}
    data::D
end
TypedVector{T}(data) where {T} = TypedVector{T,typeof(data)}(data)
@kernel coordinates(u, shape, data) = begin
    return u
end
@kernel jacobian(u, shape, data) = begin
    return 0.0
end
function ReactiveKernelsPPL.sampling_geometry(::Type{TypedVector{T}}, args, shape) where {T}
    dims = isempty(shape) ? (:(length($(args[1]))),) : shape
    ParameterGeometry(dims, :(length($(args[1]))); support=:real,
        constrain=coordinates, unconstrain=coordinates, logjac=jacobian)
end
ReactiveKernelsPPL.sampling_logdensity(::TypedVector, w) = -sum(abs2,w)/2

@testset "graph geometry with parametric constructor heads" begin
    for T in (:Float64, :(Tuple{Int,Float64})), cell in (false,true)
        data = cell ? (;x=[[1,2],[3]],y=[[0.1,0.2],[0.3]]) :
            (;x=[1,2,3],y=[0.1,0.2,0.3],key=[1,1,2])
        ast = cell ? quote
            @plate for i in eachindex(x)
                w[i] ~ TypedVector{$T}(x[i])
                y[i] .~ Normal.(w[i],1)
            end
        end : quote
            w[1:length(x)] ~ TypedVector{$T}(x)
            y .~ Normal.(w,1)
        end
        bound = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(bound)
        u = [0.2,-0.1,0.4]
        nt = constrain(built.layout,u)
        @test unconstrain(built.layout,nt) ≈ u
        @test logjac(built.layout,u) == 0
        expected = -sum(abs2,u)/2 + sum(-log(2pi)/2 .- ([0.1,0.2,0.3] .- u).^2 ./ 2)
        @test prepare_query(built,bound,:sampler)(u) ≈ expected
        source = Meta.parse(string(Base.remove_linenums!(kernel_expr(bound,built.layout))))
        replay = ReactiveKernelsPPL._eval_kernel_def(source)
        @test prepare_query((;spec=replay,layout=built.layout),bound,:sampler)(u) ≈ expected
        if !cell
            partition = prepare_partition(built,bound,:y;by=:key,locality=:error)
            @test partition_footprint(partition.kernel,u,1,nothing).unconstrained == [1,2]
        end
    end
end
end
