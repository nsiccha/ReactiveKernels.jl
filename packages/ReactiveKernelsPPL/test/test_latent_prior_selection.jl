include("fixtures/latent_prior_selection.jl")

@testset "selected prior reads retain other input consumers" begin
    fx = _prior_selection_fixture(6, 4)
    ast = Expr(:block, fx.ast.args..., :(y1 .~ Normal.(x, 1)))
    inputs = merge(fx.inputs, (; y1=zeros(9)))
    saved = deepcopy(inputs)
    bound = bind_data(lower_rkppl(ast, inputs; conditioned=(:y, :y1)), inputs)
    built = build_kernel(bound)
    @test bound.n_obs == 13
    @test built.layout.total == length(fx.u)
    sampler = prepare_sampler(built, bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    baseline = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
    _, expected = sampler_value_and_gradient!(baseline, similar(fx.u), fx.u)
    @test value ≈ fx.oracle(fx.u) + sum(logpdf.(Normal.(inputs.x, 1), inputs.y1))
    @test gradient ≈ expected
    @test inputs == saved
    bad = merge(inputs, (; y1=zeros(4)))
    # refused: the whole x likelihood still pairs nine x entries with y1.
    @test_throws "[x] column length 9 ≠ n_obs 4" bind_data(
        lower_rkppl(ast, bad; conditioned=(:y, :y1)), bad)
end

@testset "latent iterators use declared parameter axes" begin
    for prior in (:active, :derived), iterator in (:parameter, :parameter_axis),
            (n, nobs) in ((0, 4), (1, 4), (6, 4), (15, 11))
        _check_prior_selection(_prior_selection_fixture(n, nobs; prior, iterator))
    end
end

@testset "iterator counts follow Julia scalar and parameter shapes" begin
    for source in (:singleton, :anchor), value in (0.4, fill(0.4))
        inputs = (; singleton=value, y=zeros(4))
        ast = quote
            anchor ~ Normal(0, 1)
            @plate for j in eachindex($source)
                z[j] ~ Normal(anchor, 1)
            end
            y .~ Normal.(anchor + sum(z), 1)
        end
        bound = bind_data(lower_rkppl(ast, inputs; conditioned=(:y,)), inputs)
        built = build_kernel(bound)
        @test built.layout.total == 2
        u = unconstrain(built.layout, (; anchor=0.1, z=[-0.2]))
        sampler = prepare_sampler(built, bound, u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ logpdf(Normal(), 0.1) + logpdf(Normal(0.1, 1), -0.2) +
            4logpdf(Normal(-0.1, 1), 0.0)
        entries = Dict(e.name => e for e in built.layout.entries)
        @test gradient[entries[:anchor].offset] ≈ -0.1 - 0.3 + 0.4
        @test gradient[entries[:z].offset] ≈ 0.3 + 0.4
    end
    for n in (1, 6), form in (:ordered, :simplex, :matrix)
        inputs = (; domain=zeros(n), alpha=ones(n), M=zeros(n, 2), y=zeros(4))
        declaration = form === :ordered ? :(w ~ Ordered(Normal(0, 1), length(domain))) :
            form === :simplex ? :(w ~ Dirichlet(alpha)) :
            :(w[axes(M, 1), axes(M, 2)] .~ Normal.(0, 1))
        ast = quote
            anchor ~ Normal(0, 1)
            $declaration
            @plate for j in eachindex(w)
                z[j] ~ Normal(anchor, 1)
            end
            y .~ Normal.(anchor + sum(z), 1)
        end
        bound = bind_data(lower_rkppl(ast, inputs; conditioned=(:y,)), inputs)
        built = build_kernel(bound)
        values = constrain(built.layout, zeros(built.layout.total))
        @test size(values.w) == (form === :matrix ? (n, 2) : (n,))
        @test length(values.z) == length(eachindex(values.w))
        newinputs = merge(inputs, (; domain=zeros(n + 2), alpha=ones(n + 2), M=zeros(n + 2, 2)))
        rebound = build_kernel(bind_data(bound, newinputs))
        values = constrain(rebound.layout, zeros(rebound.layout.total))
        @test size(values.w) == (form === :matrix ? (n + 2, 2) : (n + 2,))
        @test length(values.z) == length(eachindex(values.w))
    end
end

@testset "latent iterators use scan, plate and opaque data extents" begin
    for iterator in (:scan, :scan_alias, :plate, :opaque_data),
            (n, nobs) in ((1, 4), (6, 4), (15, 11))
        fx = _iterator_extent_fixture(n, nobs; iterator)
        _check_prior_selection(fx)
        # Rebinding sizes from the new data, not a stored response extent.
        rebound = bind_data(fx.bound, merge(fx.inputs, (; y=zeros(nobs + 2))))
        @test build_kernel(rebound).layout.total == fx.built.layout.total
    end
    for iterator in (:plate, :opaque_data)
        _check_prior_selection(_iterator_extent_fixture(0, 4; iterator))
    end
end

@testset "rebinding reads the new iterator data" begin
    for iterator in (:parameter, :parameter_axis, :scan, :scan_alias, :plate, :opaque_data)
        fx = iterator in (:parameter, :parameter_axis) ?
            _prior_selection_fixture(6, 4; prior=:active, iterator) :
            _iterator_extent_fixture(6, 4; iterator)
        inputs = merge(fx.inputs, (; domain=zeros(9)))
        saved = deepcopy(inputs)
        rebound = bind_data(fx.bound, inputs)
        built = build_kernel(rebound)
        extra = iterator in (:parameter, :parameter_axis, :plate) ? 9 : 0
        extent = iterator === :opaque_data ? 18 : 9
        @test rebound.n_obs == 4
        @test built.layout.total == 1 + extent + extra
        @test inputs == saved
    end
end

_extent_live_calls = Ref(0)
function _extent_live_value(a, x)
    _extent_live_calls[] += 1
    return a .+ x
end

@testset "unavailable live iterator shapes never use unrelated rows" begin
    ast = quote
        anchor ~ Normal(0, 1)
        cells = _extent_live_value(anchor, domain)
        @plate for j in eachindex(cells)
            z[j] ~ Normal(cells[j], 1)
        end
        y .~ Normal.(anchor + sum(z), 1)
    end
    inputs = (; domain=zeros(6), y=zeros(4))
    saved = deepcopy(inputs)
    _extent_live_calls[] = 0
    # Capability gap: the opaque live call has no bound shape metadata.
    # 03ayeis requires a named error instead of silently allocating four
    # cells. This is not a language prohibition on such Julia values.
    error = try
        bind_data(lower_rkppl(ast, inputs; conditioned=(:y,)), inputs)
        nothing
    catch e
        e
    end
    @test error isa ContractValidationError
    @test occursin("eachindex(cells)", sprint(showerror, error))
    @test occursin("no extent established by bound data", sprint(showerror, error))
    @test _extent_live_calls[] == 0
    @test inputs == saved
    # Positive control: the same value with compiler-visible axes.
    control = deepcopy(ast)
    i = findfirst(x -> Meta.isexpr(x, :(=)) && x.args[1] === :cells, control.args)
    control.args[i] = :(cells = anchor .+ domain)
    bound = bind_data(lower_rkppl(control, inputs; conditioned=(:y,)), inputs)
    @test build_kernel(bound).layout.total == 7
end

@testset "indexed prior columns retain Julia bounds" begin
    for iterator in (:eachindex, :literal, :axes, :value)
        fx = _prior_selection_fixture(6, 4; iterator)
        for len in (0, 1, 5)
            inputs = merge(fx.inputs, (; x=zeros(len)))
            # refused: x[j] reads all six authored indices; an indexed
            # singleton does not broadcast as a shared scalar.
            @test_throws "BoundsError" begin
                bound = bind_data(lower_rkppl(fx.ast, inputs; conditioned=(:y,)), inputs)
                built = build_kernel(bound)
                query = prepare_query(built, bound, :sampler)
                Base.invokelatest(query, fx.u)
            end
        end
    end
    fx = _prior_selection_fixture(6, 4; prior=:mapped)
    inputs = merge(fx.inputs, (; index=[1,2,9,1,2,1,9,9,9]))
    # refused: a selected mapped index lies outside the two-scale domain.
    @test_throws "holds indices outside 1:2" bind_data(
        lower_rkppl(fx.ast, inputs; conditioned=(:y,)), inputs)
end

@testset "latent priors select their authored cells" begin
    for prior in (:mapped, :data, :active, :derived),
            iterator in (:eachindex, :literal, :axes, :value),
            (n, nobs) in ((0, 4), (1, 4), (6, 4), (15, 11))
        _check_prior_selection(_prior_selection_fixture(n, nobs; prior, iterator))
    end
end
