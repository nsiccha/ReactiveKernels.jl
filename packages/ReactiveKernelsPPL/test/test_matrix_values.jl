using Distributions, Test

_ma_q(layout, u) = Dict(zip(coordinate_names(layout), u))

function _ma_surface_case(kind, n)
    data = Dict{Symbol,Any}(:x1 => [0.3 + 0.13i for i in 1:n],
        :y => [0.2 + 0.05cos(i) for i in 1:n])
    x = data[:x1]
    coords = Symbol[]
    scale = q -> 1.0
    if kind == :empty
        ast = quote b[axes(X,2)] .~ Normal.(0,1); X=hcat(); mu=X*b; y .~ Normal.(mu,1) end
        loc = q -> Float64[]
    elseif kind == :empty_matrix
        ast = quote b[axes(X,2)] .~Normal.(0,1); X=hcat(x1); mu=X*b; y .~Normal.(mu,1) end
        coords=[Symbol("b.1")]
        loc = q -> Float64[]
    elseif kind == :nested
        ast = quote b[axes(Y,2)] .~ Normal.(0,1); X=hcat(ones(length(x1)),x1); Y=hcat(X,x1); mu=Y*b; y .~Normal.(mu,1) end
        coords = Symbol.("b." .* string.(1:3))
        loc = q -> [q[Symbol("b.1")] + t*(q[Symbol("b.2")]+q[Symbol("b.3")]) for t in x]
    elseif kind in (:assigned_scalar, :sampled_scalar, :literal_scalar)
        ast = kind == :assigned_scalar ?
            quote b[axes(X,2)] .~Normal.(0,1); s=1.0; X=hcat(s,x1); mu=X*b; y .~Normal.(mu,1) end :
            kind == :sampled_scalar ?
            quote b[axes(X,2)] .~Normal.(0,1); s~Normal(0,1); X=hcat(s,x1); mu=X*b; y .~Normal.(mu,1) end :
            quote b[axes(X,2)] .~Normal.(0,1); X=hcat(2,x1); mu=X*b; y .~Normal.(mu,1) end
        coords = [Symbol("b.1"),Symbol("b.2")]
        kind == :sampled_scalar && push!(coords,:s)
        loc = q -> [(kind == :sampled_scalar ? q[:s] : kind == :literal_scalar ? 2.0 : 1.0)*q[Symbol("b.1")] + t*q[Symbol("b.2")] for t in x]
    elseif kind in (:inline, :inline_location)
        ast = kind == :inline ?
            quote b[1:2] .~Normal.(0,1); mu=hcat(ones(length(x1)),x1)*b; y .~Normal.(mu,1) end :
            quote b[1:2] .~Normal.(0,1); y .~Normal.(hcat(ones(length(x1)),x1)*b,1) end
        coords = [Symbol("b.1"),Symbol("b.2")]
        loc = q -> [q[Symbol("b.1")]+t*q[Symbol("b.2")] for t in x]
    elseif kind in (:unused, :unused_b)
        ast = kind == :unused ?
            quote X=hcat(ones(length(x1)),x1); mu=a.+c.*x1; a~Normal(0,1); c~Normal(0,1); y .~Normal.(mu,1) end :
            quote b[axes(X,2)] .~Normal.(0,1); X=hcat(ones(length(x1)),x1); mu=a.+c.*x1; a~Normal(0,1); c~Normal(0,1); y .~Normal.(mu,1) end
        coords=[:a,:c]
        kind==:unused_b && append!(coords,[Symbol("b.1"),Symbol("b.2")])
        loc = q ->[q[:a]+q[:c]*t for t in x]
    elseif kind == :data_product
        ast = quote X=hcat(ones(length(x1)),x1); mu=X*x1; y .~Normal.(mu,1) end
        loc = q ->[x[1]+t*x[2] for t in x]
    elseif kind in (:scalar_s, :scalar_b)
        ast = kind == :scalar_s ?
            quote s~Normal(0,1); X=hcat(ones(length(x1)),x1); mu=X*s; y .~Normal.(mu,1) end :
            quote b~Normal(0,1); X=hcat(ones(length(x1)),x1); mu=X*b; y .~Normal.(mu,1) end
        nm = kind == :scalar_s ? :s : :b
        coords=[nm]
        loc = q ->[j==1 ? q[nm] : x[i]*q[nm] for i in 1:n,j in 1:2]
    elseif kind == :matrix_product
        data[:x2]=[0.7+0.2sin(i) for i in 1:n]
        z=data[:x2]
        ast=quote X=hcat(ones(length(x1)),x1); Y=hcat(x2); y .~Normal.(X*Y,1) end
        loc = q ->reshape([z[1]+t*z[2] for t in x],:,1)
    elseif kind in (:literal_product, :named_literal_product)
        ast=kind == :literal_product ?
            quote X=hcat(ones(length(x1)),x1); y .~Normal.(X*2,1) end :
            quote X=hcat(ones(length(x1)),x1); mu=X*2; y .~Normal.(mu,1) end
        loc = q ->[j==1 ? 2.0 : 2x[i] for i in 1:n,j in 1:2]
    elseif kind in (:raw_location, :alias)
        ast=kind == :raw_location ?
            quote X=hcat(ones(length(x1)),x1); y .~Normal.(X,1) end :
            quote X=hcat(ones(length(x1)),x1); mu=X; y .~Normal.(mu,1) end
        loc = q ->[j==1 ? 1.0 : x[i] for i in 1:n,j in 1:2]
    elseif kind in (:plus_scalar, :alias_plus_scalar)
        ast=kind == :plus_scalar ?
            quote X=hcat(ones(length(x1)),x1); mu=a.+X; a~Normal(0,1); y .~Normal.(mu,1) end :
            quote X=hcat(ones(length(x1)),x1); w=X; mu=a.+w; a~Normal(0,1); y .~Normal.(mu,1) end
        coords=[:a]
        loc = q ->[q[:a]+(j==1 ? 1.0 : x[i]) for i in 1:n,j in 1:2]
    elseif kind == :broadcast_b
        ast=quote b[axes(X,2)] .~Normal.(0,1); X=hcat(ones(length(x1)),x1); mu=X.*b; y .~Normal.(mu,1) end
        coords=[Symbol("b.1"),Symbol("b.2")]
        # Julia broadcasts a vector along rows, rather than scaling columns.
        loc = q ->[q[Symbol("b.$i")]*(j==1 ? 1.0 : x[n==1 ? 1 : i]) for i in 1:2,j in 1:2]
    elseif kind == :matrix_scale
        ast=quote b[axes(X,2)] .~Normal.(0,1); X=hcat(ones(length(x1)),x1); mu=X*b; y .~Normal.(mu,X) end
        coords=[Symbol("b.1"),Symbol("b.2")]
        loc = q ->[q[Symbol("b.1")]+t*q[Symbol("b.2")] for t in x]
        scale = q ->[j==1 ? 1.0 : x[i] for i in 1:n,j in 1:2]
    elseif kind in (:live_column, :live_column_axes, :live_column_matrix)
        # Data columns concatenated with a parameter-dependent vector column.
        ast = kind == :live_column ?
            quote a~Normal(0,1); v=exp.(a.*x1); X=hcat(ones(length(x1)),x1,v); b[1:3] .~Normal.(0,1); y .~Normal.(X*b,1.0) end :
            kind == :live_column_axes ?
            quote a~Normal(0,1); v=exp.(a.*x1); X=hcat(ones(length(x1)),x1,v); b[axes(X,2)] .~Normal.(0,1); y .~Normal.(X*b,1.0) end :
            quote a~Normal(0,1); D=hcat(ones(length(x1)),x1); X=hcat(D,exp.(a.*x1)); b[1:3] .~Normal.(0,1); y .~Normal.(X*b,1.0) end
        coords=[:a,Symbol("b.1"),Symbol("b.2"),Symbol("b.3")]
        loc = q ->[q[Symbol("b.1")]+t*q[Symbol("b.2")]+exp(q[:a]*t)*q[Symbol("b.3")] for t in x]
    else
        error("unknown matrix fixture $kind")
    end
    return (; kind,n,data,ast,coords,loc,scale)
end

function _ma_surface_parts(case, layout, u)
    q=_ma_q(layout,u)
    prior=sum(logpdf(Normal(0,1),v) for v in u; init=0.0)
    likelihood=sum(logpdf.(Normal.(case.loc(q),case.scale(q)),case.data[:y]))
    return (;prior,likelihood,jac=0.0,posterior=prior+likelihood)
end

function _ma_surface_model(case)
    plan=lower_rkppl(case.ast,Set(keys(case.data));conditioned=(:y,))
    bound=bind_data(plan,case.data)
    return bound,build_kernel(bound)
end

function _ma_surface_check(case)
    original=deepcopy(case.data)
    bound,built=_ma_surface_model(case)
    @test Set(coordinate_names(built.layout))==Set(case.coords)
    u=[0.2sin(i) for i in 1:built.layout.total]
    oracle=w->_ma_surface_parts(case,built.layout,w).posterior
    for shift in (isempty(u) ? (0.0,) : (0.0,0.17))
        v=u.+shift
        parts=_ma_surface_parts(case,built.layout,v)
        @test logjac(built.layout,v)==0.0
        @test unconstrain(built.layout,constrain(built.layout,v))≈v
        @test _query(built.spec,bound,:prior,v)≈parts.prior
        @test _query(built.spec,bound,:likelihood,v)≈parts.likelihood
        _check_model_math(built,bound,v,oracle)
    end
    @test isequal(case.data,original)
    return (;case,bound,built,u,oracle)
end

const _MA_SURFACE_FIXTURES=(
    (:empty_matrix,0),(:nested,5),(:assigned_scalar,1),(:sampled_scalar,1),
    (:literal_scalar,1),(:inline,5),(:inline_location,5),(:unused,5),
    (:unused_b,5),(:data_product,2),(:scalar_s,5),(:scalar_b,5),
    (:matrix_product,2),(:literal_product,5),(:named_literal_product,5),
    (:raw_location,5),(:alias,5),(:plus_scalar,5),(:alias_plus_scalar,5),
    (:broadcast_b,1),(:broadcast_b,2),(:matrix_scale,5),
    (:live_column,1),(:live_column,5),(:live_column_axes,5),
    (:live_column_matrix,5))

@testset "ordinary matrix values: independent math and native reverse" begin
    for (kind,n) in _MA_SURFACE_FIXTURES
        @testset "$kind / $n" begin
            _ma_surface_check(_ma_surface_case(kind,n))
        end
    end
end

@testset "ordinary matrix values: original dimension mismatches" begin
    for n in (3,6),kind in (:empty,:assigned_scalar,:sampled_scalar,:literal_scalar,
            :data_product,:matrix_product,:broadcast_b)
        case=_ma_surface_case(kind,n)
        # Preserve the historical invalid dimensions. The same AST has a
        # valid control above, except hcat(): Base returns a vector, so its
        # distinct empty-matrix control explicitly supplies an empty column.
        err=try
            bound,built=_ma_surface_model(case)
            _query(built.spec,bound,:posterior,fill(0.2,built.layout.total))
            nothing
        catch e
            e
        end
        @test err isa DimensionMismatch || err isa ContractValidationError
        @test occursin(kind==:empty ? r"vector times a vector|not a product|DimensionMismatch" : r"DimensionMismatch|different row counts",sprint(showerror,err))
        # Verify the underlying Julia failure independently of the lowering.
        x=case.data[:x1]
        X=hcat(ones(n),x)
        err=try
            kind==:empty ? hcat()*[0.2] :
            kind in (:assigned_scalar,:sampled_scalar) ? hcat(0.2,x) :
            kind==:literal_scalar ? hcat(2,x) :
            kind==:data_product ? X*x :
            kind==:matrix_product ? X*hcat(case.data[:x2]) :
            X.*[0.2,0.3]
            nothing
        catch e
            e
        end
        @test err isa DimensionMismatch || err isa MethodError
    end
end

function _ma_unused_columns(n)
    fixtures=Pair{Symbol,Any}[
        :X=>ones(n+1,2),:X=>fill("a",n,2),:X=>ones(n,0),
        :X=>reshape(Union{Missing,Float64}[missing;ones(2n-1)],n,2),
        :X=>ones(n,2,2),:c=>1.5]
    return [(name,merge(Dict{Symbol,Any}(_columns(n)),Dict(name=>value)))
        for (name,value) in fixtures]
end

@testset "empty hcat follows Base" begin
    @test hcat() isa Vector
    @test size(hcat(),2)==1
end


function _ma_role_plan(kind,n)
    cols=Dict{Symbol,Any}(_columns(n))
    X=hcat(ones(n),collect(1.0:n))
    cols[:X]=X
    plan=_unbind(_gaussian_plan(n))
    if kind==:response
        plan.responses[1]=LikelihoodSpec(GaussianFam,IdentityLink,:X,:mu,:sigma,nothing,_none_evidence(),:y_resp)
    elseif kind==:term
        plan.predictors[1]=PredictorSpec(:mu,IdentityLink,TermSpec[
            TermSpec(InterceptTerm,ColumnRef[],NamedTuple(),:Intercept,:intercept),
            TermSpec(ContinuousTerm,[:X],NamedTuple(),:X,:x_term)],:mu)
        plan.population_priors[2]=PopulationPrior(:mu,:X,0.0,1.0)
    elseif kind==:weights
        plan.responses[1]=LikelihoodSpec(GaussianFam,IdentityLink,:y,:mu,:sigma,:X,_none_evidence(),:y_resp)
    elseif kind==:scale
        plan.responses[1]=LikelihoodSpec(GaussianFam,IdentityLink,:y,:mu,:X,nothing,_none_evidence(),:y_resp)
    elseif kind==:group
        plan=StructuralPlan(plan.responses,plan.predictors,plan.population_priors,
            plan.parameters,plan.assignments,Dict{Symbol,ColumnData}(),0;
            levelmaps=[LevelMap(:mu,:X,[],:levels,Colon())])
    else
        error("unknown matrix role $kind")
    end
    return plan,cols
end

function _ma_role_parts(kind,data,u)
    sigma=exp(u[3])
    mu=u[1].+u[2].*(kind==:term ? data[:X] : data[:x])
    response=kind==:response ? data[:X] : data[:y]
    scale=kind==:scale ? data[:X] : sigma
    weights=kind==:weights ? data[:X] : 1.0
    likelihood=sum(weights.*logpdf.(Normal.(mu,scale),response))
    prior=logpdf(Normal(0,1),u[1])+logpdf(Normal(0,1),u[2])+logpdf(Exponential(1),sigma)
    return (;prior,likelihood,jac=u[3],posterior=prior+likelihood+u[3])
end

@testset "matrix values in hand-authored roles: independent math" begin
    for n in (9,13),kind in (:response,:term,:weights,:scale,:group)
        @testset "$kind / $n" begin
            plan,data=_ma_role_plan(kind,n)
            original=deepcopy(data)
            bound=bind_data(plan,data)
            built=build_kernel(bound)
            @test built.layout.total==3
            kind==:group && @test only(bound.levelmaps).values==collect(1.0:n)
            for u in ([0.3,-0.2,0.1],[-0.1,0.25,-0.2])
                parts=_ma_role_parts(kind,data,u)
                @test logjac(built.layout,u)≈parts.jac
                @test _query(built.spec,bound,:prior,u)≈parts.prior
                @test _query(built.spec,bound,:likelihood,u)≈parts.likelihood
                _check_model_math(built,bound,u,w->_ma_role_parts(kind,data,w).posterior)
            end
            @test isequal(data,original)
        end
    end
end

@testset "hand-authored plans ignore requirements on all six unused values" begin
    for (name,data) in _ma_unused_columns(9)
        original=deepcopy(data)
        bound=bind_data(_unbind(_gaussian_plan(9)),data)
        built=build_kernel(bound)
        @test bound.n_obs==9
        @test isequal(bound.columns[name],data[name])
        @test built.layout.total==3
        for u in ([0.3,-0.2,0.1],[-0.1,0.25,-0.2])
            parts=_ma_role_parts(:unused,data,u)
            @test logjac(built.layout,u)≈parts.jac
            @test _query(built.spec,bound,:prior,u)≈parts.prior
            @test _query(built.spec,bound,:likelihood,u)≈parts.likelihood
            _check_model_math(built,bound,u,w->_ma_role_parts(:unused,data,w).posterior)
        end
        @test isequal(data,original)
    end
end


@rkppl _ma_stream(mu,sigma)=begin
    y .~Normal.(mu,sigma)
    return y
end

@testset "unused raw data keep no observation shape or numerical requirements" begin
    for unused in (ones(10,2),fill("a",9,2),ones(9,0),
            reshape(Union{Missing,Float64}[missing;ones(17)],9,2),
            ones(9,2,2),1.5)
        case=_ma_surface_case(:unused,9)
        case.data[:extra]=unused
        _ma_surface_check(case)
    end
end

@testset "matrix location through a submodel keeps the unused coefficient prior" begin
    case=_ma_surface_case(:raw_location,5)
    ast=quote
        b[axes(X,2)] .~Normal.(0,1)
        X=hcat(ones(length(x1)),x1)
        mu=X*b
        y~_ma_stream(X,1.0)
    end
    case=merge(case,(;ast,coords=[Symbol("b.1"),Symbol("b.2")]))
    _ma_surface_check(case)
end
