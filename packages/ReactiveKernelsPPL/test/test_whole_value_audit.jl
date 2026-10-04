using Test, Distributions, ReactiveKernelsPPL
using ReactiveKernels
using DifferentiationInterface: AutoEnzyme
import Enzyme

module RetainedModuleValueAudit
using ReactiveKernelsPPL
@inline vector_shift(v, b) = v .+ b[1] .+ b[2] .* v
@inline scalar_shift(v, b) = b[1] + b[2] * sum(v)
@inline value_scale(v) = sqrt.(1 .+ v.^2)
end

@testset "retained module values: normalized density, full Reverse, source replay" begin
    for name in (:loc, :mu), scalar in (false, true), inline_scale in (false, true)
        fn = scalar ? :scalar_shift : :vector_shift
        scale_stmt = inline_scale ? "" : "sigma = value_scale($name)"
        scale_arg = inline_scale ? "value_scale($name)" : "sigma"
        source = """
        @rkppl begin
            b[1:2] .~ Normal.(0.0, 1.0)
            $name = $fn(x, b)
            $scale_stmt
            y .~ Normal.($name, $scale_arg)
        end
        """
        authored = Meta.parse(source)
        replay = sprint(io -> Base.show_unquoted(io, authored))
        for ast in (authored, Meta.parse(replay))
            model = Core.eval(RetainedModuleValueAudit, ast)
            ast_before = deepcopy(model.ast)
            for n in (2, 7, 19)
                data = (; x=collect(range(-0.4, 0.6; length=n)),
                    y=collect(range(0.1, 0.8; length=n)))
                before = deepcopy(data)
                bound = model(; x=data.x) | (; y=data.y)
                built = build_kernel(bound)
                @test coordinate_names(built.layout) == [Symbol("b.1"), Symbol("b.2")]
                sampler = prepare_sampler(built, bound, zeros(2);
                    backend=AutoEnzyme(; mode=Enzyme.Reverse))
                for q in ([0.0, 0.0], [-0.35, 0.2], [0.25, -0.15])
                    q_before = copy(q)
                    gradient = fill(NaN, 2)
                    density, _ = sampler_value_and_gradient!(sampler, gradient, q)
                    # Independent normalized prior and Normal likelihood,
                    # including the scale's dependence on the mean.
                    expected = -log(2pi) - 0.5sum(abs2, q)
                    expected_gradient = -copy(q)
                    for i in eachindex(data.x, data.y)
                        slope = scalar ? sum(data.x) : data.x[i]
                        mu = (scalar ? 0.0 : data.x[i]) + q[1] + q[2] * slope
                        variance = 1 + mu^2
                        residual = data.y[i] - mu
                        expected += -0.5log(2pi) - 0.5log(variance) -
                            0.5residual^2 / variance
                        dm = residual / variance + mu *
                            (residual^2 / variance^2 - 1 / variance)
                        expected_gradient[1] += dm
                        expected_gradient[2] += slope * dm
                    end
                    @test density ≈ expected rtol=1e-12 atol=1e-12
                    @test gradient ≈ expected_gradient rtol=1e-12 atol=1e-12
                    @test isequal(q, q_before)
                    @test isequal(data, before)
                end
                @test isequal(model.ast, ast_before)
            end
        end
    end
end

function _wva_flat_literal_fixture(dotted, form, n=3; replay=false)
    literal = dotted ? "[a, b, a .+ b]" : "[a, b, a+b]"
    definition = form === :inline ? "" : "mu = $literal"
    location = form === :inline ? literal :
        form === :alias ? "location" : form === :gather ? "mu[rows]" : "mu"
    alias = form === :alias ? "location = mu" : ""
    source = """
    module FlatLiteralLocationAudit
        using ReactiveKernelsPPL
        const model = @rkppl begin
            a ~ Normal(0.0, 1.0)
            b ~ Normal(0.0, 1.0)
            $definition
            $alias
            y .~ Normal.($location, 1.0)
        end
    end
    """
    ast = Meta.parse(source)
    if replay
        ast = Meta.parse(sprint(io -> Base.show_unquoted(io, ast)))
    end
    # Evaluate the complete defining module in a fresh namespace. Replay
    # keeps both the literal syntax and its producer module context.
    box = Module(gensym(:FlatLiteralEvaluation))
    Core.eval(box, ast)
    model = getfield(getfield(box, :FlatLiteralLocationAudit), :model)
    rows = form === :gather ? [mod1(2i, 3) for i in 1:n] : [1, 2, 3]
    data = form === :gather ?
        (; rows, y=collect(range(-0.7, 0.6; length=n))) :
        (; y=[-0.7, 0.2, 0.6])
    bound = (form === :gather ? model(; rows=data.rows) : model()) | (; y=data.y)
    built = build_kernel(bound)
    function oracle(q)
        result = -log(2pi) - 0.5sum(abs2, q)
        for i in eachindex(rows, data.y)
            mean = rows[i] == 1 ? q[1] : rows[i] == 2 ? q[2] : q[1] + q[2]
            result += -0.5log(2pi) - 0.5(data.y[i] - mean)^2
        end
        return result
    end
    function gradient(q)
        result = -copy(q)
        for i in eachindex(rows, data.y)
            mean = rows[i] == 1 ? q[1] : rows[i] == 2 ? q[2] : q[1] + q[2]
            residual = data.y[i] - mean
            rows[i] != 2 && (result[1] += residual)
            rows[i] != 1 && (result[2] += residual)
        end
        return result
    end
    return (; model, bound, built, data, oracle, gradient)
end

@testset "flat scalar vector literals: full density, Reverse and defining-module replay" begin
    for dotted in (false, true), form in (:named, :inline, :alias, :gather),
            replay in (false, true), n in (form === :gather ? (2, 7, 19) : (3,))
        fixture = _wva_flat_literal_fixture(dotted, form, n; replay)
        (; model, bound, built, data, oracle, gradient) = fixture
        data_before, ast_before = deepcopy(data), deepcopy(model.ast)
        @test coordinate_names(built.layout) == [:a, :b]
        sampler = prepare_sampler(built, bound, zeros(2);
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for q in ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.7])
            before = copy(q)
            actual_gradient = fill(NaN, 2)
            value, _ = sampler_value_and_gradient!(sampler, actual_gradient, q)
            @test value ≈ oracle(q) atol=2e-11 rtol=2e-11
            @test actual_gradient ≈ gradient(q) atol=2e-10 rtol=2e-10
            @test isequal(q, before)
            @test isequal(data, data_before)
        end
        @test isequal(model.ast, ast_before)
    end
end

# The existing array-value fixtures supply _adv_model and the ordinary leaf
# identity in ArrayDataValueModels. Oracles below unpack coordinates by their
# authored names and implement every transform independently of the layout.
_wva_coordinates(u, names) = Dict(zip(names, u))
_wva_posterior(parts) = parts.prior + parts.likelihood + parts.jac

function _wva_build(case; values=false)
    plan = lower_rkppl(case.ast, values ? case.data : Set(keys(case.data));
        mod=case.mod, conditioned=keys(case.data))
    bound = bind_data(plan, case.data)
    return bound, build_kernel(bound)
end

function _wva_check(case; values=false)
    before = deepcopy(case.data)
    ast_before = deepcopy(case.ast)
    bound, built = _wva_build(case; values)
    names = coordinate_names(built.layout)
    @test Set(names) == Set(case.coords)
    @test length(names) == length(case.coords)
    u = [0.15cos(i) for i in eachindex(names)]
    oracle = w -> _wva_posterior(case.parts(_wva_coordinates(w, names)))
    for shift in (0.0, 0.17)
        v = u .+ shift
        parts = case.parts(_wva_coordinates(v, names))
        @test _query(built.spec, bound, :prior, v) ≈ parts.prior atol=1e-11
        @test _query(built.spec, bound, :likelihood, v) ≈ parts.likelihood atol=1e-11
        @test _query(built.spec, bound, :log_jacobian, v) ≈ parts.jac atol=1e-12
        @test logjac(built.layout, v) ≈ parts.jac atol=1e-12
        @test unconstrain(built.layout, constrain(built.layout, v)) ≈ v atol=1e-12
        _check_model_math(built, bound, v, oracle)
    end
    @test isequal(case.data, before)
    @test isequal(case.ast, ast_before)
    return (;case, bound, built, u, oracle)
end

const _WVA_EXTRAS = (
    quote mu = r[g] .+ gx; y .~ Normal.(mu, sigma) end,
    :(y .~ weighted.(Normal.(r[g], sigma), gx)),
    quote y .~ Normal.(r[g], sigma); y2 .~ Normal.(gx, sigma) end,
    quote y .~ Normal.(r[g], sigma); u = gx .+ 1; y2 .~ Normal.(u, sigma) end)

function _wva_array_case(kind, variant, K, n; second_rows=K)
    ast = _adv_model(kind, :named)
    pop!(ast.args)
    extra = _WVA_EXTRAS[variant]
    append!(ast.args, extra.head === :block ? extra.args : Any[extra])
    data = Dict{Symbol,Any}(_adv_data(K,n))
    data[:g] = [mod1(j+1,K) for j in 1:n]
    data[:gx] = [0.3+0.05cos(j) for j in 1:K]
    variant >= 3 && (data[:y2] = [0.1sin(j) for j in 1:second_rows])
    array_names = kind === :vector ? [Symbol("z.$j") for j in 1:K] :
        [Symbol("Z.$j.$k") for k in 1:2 for j in 1:K]
    coords = [:a, :b, :sigma, array_names...]
    function parts(q)
        z = kind === :vector ? [q[Symbol("z.$j")] for j in 1:K] :
            [q[Symbol("Z.$j.1")] for j in 1:K]
        sigma = exp(q[:sigma])
        r = q[:a] .+ z .+ q[:b] .* data[:gx]
        location = variant == 1 ? r[data[:g]] .+ data[:gx] : r[data[:g]]
        weights = variant == 2 ? data[:gx] : 1.0
        likelihood = sum(weights .* logpdf.(Normal.(location,sigma), data[:y]))
        if variant >= 3
            second = variant == 4 ? data[:gx] .+ 1 : data[:gx]
            likelihood += sum(logpdf.(Normal.(second,sigma), data[:y2]))
        end
        prior = logpdf(Normal(),q[:a]) + logpdf(Normal(),q[:b]) +
            logpdf(Exponential(1),sigma) + sum(logpdf(Normal(),q[name]) for name in array_names)
        return (;prior,likelihood,jac=q[:sigma])
    end
    return (;label="whole/$kind/$variant/$K/$n", ast,data,coords,parts,mod=ArrayDataValueModels)
end

function _wva_affine_case(n=3, ny2=2)
    ast = quote
        z[1:2] .~ Normal.(0,1)
        value = z .+ group_x
        reads = ArrayDataValueModels.passthrough(value)
        y .~ Normal.(reads[obs_index],1)
        y2 .~ Normal.(group_x,1)
    end
    data = Dict{Symbol,Any}(:group_x=>[0.2,-0.4],
        :obs_index=>[mod1(i+1,2) for i in 1:n],
        :y=>[0.1sin(i) for i in 1:n],:y2=>[0.1cos(i) for i in 1:ny2])
    coords = [Symbol("z.1"),Symbol("z.2")]
    parts(q) = begin
        z = [q[Symbol("z.1")],q[Symbol("z.2")]]
        reads = z .+ data[:group_x]
        prior = sum(logpdf.(Normal(),z))
        likelihood = sum(logpdf.(Normal.(reads[data[:obs_index]],1),data[:y])) +
            sum(logpdf.(Normal.(data[:group_x],1),data[:y2]))
        (;prior,likelihood,jac=0.0)
    end
    return (;label="affine/$n/$ny2",ast,data,coords,parts,mod=Main)
end

function _wva_vector_case(kind,n=2)
    if kind === :simplex
        ast = quote a ~ Normal(0,5); phi ~ Dirichlet([1.,1.]);
            sigma ~ Exponential(1); mu = a .+ phi .* x1; y .~ Normal.(mu,sigma) end
        data = Dict{Symbol,Any}(:x1=>[0.3+0.1i for i in 1:n],:y=>[0.2sin(i) for i in 1:n])
        coords = [:a,:sigma,Symbol("phi.1")]
        parts = q -> begin
            f = 1/(1+exp(-q[Symbol("phi.1")]))
            phi = [f,1-f]; sigma = exp(q[:sigma])
            prior = logpdf(Normal(0,5),q[:a]) + logpdf(Dirichlet([1.,1.]),phi) +
                logpdf(Exponential(1),sigma)
            likelihood = sum(logpdf.(Normal.(q[:a] .+ phi .* data[:x1],sigma),data[:y]))
            (;prior,likelihood,jac=q[:sigma]+log(f)+log1p(-f))
        end
    else
        ast = quote c ~ Ordered(Normal(0,1),2); mu = c .* x; y2 .~ Normal.(mu,1) end
        data = Dict{Symbol,Any}(:x=>[0.3+0.1i for i in 1:n],:y2=>[0.2sin(i) for i in 1:n])
        coords = [Symbol("c.1"),Symbol("c.2")]
        parts = q -> begin
            c = [q[Symbol("c.1")],q[Symbol("c.1")]+exp(q[Symbol("c.2")])]
            prior = sum(logpdf.(Normal(),c))
            likelihood = sum(logpdf.(Normal.(c .* data[:x],1),data[:y2]))
            (;prior,likelihood,jac=q[Symbol("c.2")])
        end
    end
    return (;label=String(kind)*"/$n",ast,data,coords,parts,mod=Main)
end

function _wva_scalar_case(n=7, s=0.7)
    ast = quote a ~ Normal(0,5); b ~ Normal(0,2); mu = a .+ b .* x;
        y .~ Normal.(mu,s) end
    data = Dict{Symbol,Any}(:x=>[0.5cos(i) for i in 1:n],
        :y=>[0.2sin(i) for i in 1:n],:s=>s)
    coords = [:a,:b]
    parts(q) = (;prior=logpdf(Normal(0,5),q[:a])+logpdf(Normal(0,2),q[:b]),
        likelihood=sum(logpdf.(Normal.(q[:a] .+ q[:b] .* data[:x],s),data[:y])),jac=0.0)
    return (;label="scalar/$n/$s",ast,data,coords,parts,mod=Main)
end

function _wva_replacement_case(g=1,n=7)
    base = @rkppl begin
        a ~ Normal(0,1)
        c[levels(g)[2:end]] .~ Normal.(0,2)
        mu = a .+ c[g]
        y .~ Normal.(mu,1.5)
    end
    replacement = Base.merge(base, :(c ~ Normal(0,1)))
    # Rewriting the declaration preserves the original body and indexing.
    ast = replacement.ast
    data = Dict{Symbol,Any}(:g=>g,:y=>[0.1sin(i) for i in 1:n])
    coords = [:a,:c]
    parts(q) = (;prior=logpdf(Normal(),q[:a])+logpdf(Normal(),q[:c]),
        likelihood=sum(logpdf.(Normal.(q[:a] .+ getindex(q[:c],g),1.5),data[:y])),jac=0.0)
    return (;label="replacement/$g/$n",ast,data,coords,parts,mod=Main)
end

function _wva_weibull_case(form=:retained,n=7)
    ast = quote a ~ Normal(0,1); b ~ Normal(0,1); eta = a .+ b .* x end
    form != :inline && push!(ast.args, :(theta = 0.8 .+ abs.(eta)))
    form === :shared && push!(ast.args, :(y2 .~ Normal.(eta,1.0)))
    push!(ast.args,form === :named ? :(y .~ Weibull.(1.4,theta)) :
        :(y .~ Weibull.(1.4,0.8 .+ abs.(eta))))
    data = Dict{Symbol,Any}(:x=>[0.4cos(i) for i in 1:n],:y=>[0.8+0.3sin(i) for i in 1:n])
    form === :shared && (data[:y2] = [0.2cos(i) for i in 1:n])
    coords = [:a,:b]
    parts(q) = (;prior=logpdf(Normal(),q[:a])+logpdf(Normal(),q[:b]),
        likelihood=sum(logpdf.(Weibull.(1.4,0.8 .+ abs.(q[:a] .+ q[:b] .* data[:x])),data[:y])) +
            (form === :shared ? sum(logpdf.(Normal.(q[:a] .+ q[:b] .* data[:x],1),data[:y2])) : 0.0),jac=0.0)
    return (;label="Weibull/$form/$n",ast,data,coords,parts,mod=Main)
end

function _wva_literal_case(source=:literal,location=0.8,n=7)
    ex = source === :literal ? :([location,a1]) : :(vcat(location,a1))
    ast = quote
        a1 ~ Exponential(1)
        a ~ Normal(0,1)
        @plate for i in eachindex(y)
            sd = $ex[g[i]]
            y[i] ~ Normal(a,sd)
        end
    end
    width = location isa Number ? 2 : 3
    data = Dict{Symbol,Any}(:location=>location,:g=>[mod1(i,width) for i in 1:n],
        :y=>[0.2sin(i) for i in 1:n])
    coords = [:a1,:a]
    parts(q) = begin
        a1 = exp(q[:a1])
        values = source === :literal ? [location,a1] : vcat(location,a1)
        prior = logpdf(Exponential(1),a1)+logpdf(Normal(),q[:a])
        likelihood = sum(logpdf(Normal(q[:a],values[i]),y) for (i,y) in zip(data[:g],data[:y]))
        (;prior,likelihood,jac=q[:a1])
    end
    return (;label="literal/$source/$(typeof(location))/$n",ast,data,coords,parts,mod=Main)
end

@testset "whole-array audit: all eight original uses with independent math" begin
    for kind in (:vector,:column),variant in 1:4
        _wva_check(_wva_array_case(kind,variant,3,3))
        variant >= 3 && _wva_check(_wva_array_case(kind,variant,3,7))
    end
    _wva_check(_wva_affine_case())
end

@testset "whole-vector arithmetic, names-only binding and unchanged scalar indexing" begin
    _wva_check(_wva_vector_case(:simplex))
    _wva_check(_wva_vector_case(:ordered))
    for s in (0.7,1.1), door in (false,true)
        _wva_check(_wva_scalar_case(7,s);values=door)
    end
    _wva_check(_wva_replacement_case())
end

@testset "retained values and literal gathers: independent math" begin
    for form in (:retained,:named,:inline,:shared)
        _wva_check(_wva_weibull_case(form))
    end
    for source in (:literal,:vcat)
        _wva_check(_wva_literal_case(source,0.8))
    end
    _wva_check(_wva_literal_case(:vcat,[0.8,0.9]))
end

function _wva_failure(case)
    try
        bound,built = _wva_build(case)
        _query(built.spec,bound,:posterior,fill(0.15,built.layout.total))
        nothing
    catch e
        e
    end
end

@testset "whole-value audit preserves original Julia invalid dimensions and indices" begin
    for kind in (:vector,:column),variant in 1:4
        case = _wva_array_case(kind,variant,3,7;second_rows=7)
        @test _wva_failure(case) isa Union{DimensionMismatch,ContractValidationError}
        @test_throws DimensionMismatch case.parts(Dict(name=>0.15 for name in case.coords))
    end
    case = _wva_affine_case(3,3)
    @test _wva_failure(case) isa Union{DimensionMismatch,ContractValidationError}
    @test_throws DimensionMismatch [0.2,-0.4] .+ ones(3)
    for kind in (:simplex,:ordered)
        case = _wva_vector_case(kind,5)
        @test _wva_failure(case) isa Union{DimensionMismatch,ContractValidationError}
        @test_throws DimensionMismatch ones(2) .* ones(5)
    end
    for g in ([1,1,1],[1,2,1],2)
        case = _wva_replacement_case(g)
        @test _wva_failure(case) isa (g isa Vector ? MethodError : BoundsError)
        @test_throws (g isa Vector ? MethodError : BoundsError) getindex(0.7,g)
    end
    # [vector, scalar] is a nested container; a separate authored vcat
    # control above concatenates it. The literal is never silently flattened.
    case = _wva_literal_case(:literal,[0.8,0.9])
    @test _wva_failure(case) isa Union{MethodError,BoundsError,ContractValidationError}
    @test [case.data[:location],0.7][1] === case.data[:location]
    @test_throws MethodError Normal(0.0,[case.data[:location],0.7][1])
end
