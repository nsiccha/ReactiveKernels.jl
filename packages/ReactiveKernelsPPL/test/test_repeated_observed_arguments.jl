module TestRepeatedObservedArguments

using Test, ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels, Enzyme
using ReactiveKernels.DifferentiationInterface: AutoEnzyme

@rkppl channel(y, shift) = begin
    offset ~ Normal(0, 1)
    loc = offset + shift
    y .~ Normal.(loc, 1)
    return loc
end

@rkppl nested(y, shift) = begin
    offset ~ Normal(0, 1)
    @plate for i in eachindex(y)
        loc[i] = shift[i] + offset
        width[i] = 1
    end
    y .~ Normal.(loc, width)
    return loc
end

@rkppl scalar_channel(y, shift) = begin
    offset ~ Normal(0, 1)
    loc = offset + shift
    y ~ Normal(loc, 1)
    return loc
end

normal_logpdf(y, mu) = -0.5 * ((y - mu)^2 + log(2pi))
sum_cell(x::Number) = x
sum_cell(x) = sum(x; init=0.0)
nodes(p::ReactiveKernels._PlateProgram) = nodes(p.ast)
nodes(ex::Expr) = 1 + sum(nodes, ex.args; init=0)
nodes(ex) = 1
loops(p::ReactiveKernels._PlateProgram) = loops(p.ast)
loops(ex::Expr) = (ex.head in (:for, :while)) + sum(loops, ex.args; init=0)
loops(ex) = 0

@testset "A direct submodel observes its argument" begin
    data = (; values=[0.1, 0.4])
    plan = bind_data(lower_rkppl(:(begin member ~ channel(values, 0.5) end), data;
        mod=@__MODULE__, conditioned=(:values,)), data)
    built = build_kernel(plan)
    u = [0.2]
    @test built.layout.total == 1
    @test coordinate_names(built.layout) == [Symbol("member.offset")]
    @test constrain(built.layout, u).member.offset == 0.2
    query(w) = Base.invokelatest(prepare_query(built, plan, w), u)
    expected = normal_logpdf.(data.values, 0.7)
    @test query(:pointwise).values ≈ expected
    @test query(:likelihood) ≈ sum(expected)
    @test query(:prior) ≈ normal_logpdf(0.2, 0.0)
    @test query(:sampler) ≈ sum(expected) + normal_logpdf(0.2, 0.0)
end

@testset "Repeated submodels observe top-level arguments" begin
    for (callee, scalar, nested_locations) in
            ((:channel, false, false), (:nested, false, true), (:scalar_channel, true, false))
        model = quote
            @plate for e in eachindex(values)
                member[e] ~ $callee(values[e], shift)
            end
            fitted = member.loc
        end
        programs = Expr[]
        structures = Any[]
        emitted = Tuple{Int,Int}[]
        reference_plan = nothing
        for n in (2, 7, 0)
            readings = [[0.1 * i - 0.4 for i in 1:(e % 4)] for e in 1:n]
            values = scalar ? [0.1 * e for e in 1:n] : readings
            shift = nested_locations ? [0.3, 0.4, -0.2] : 0.5
            data = (; values, shift)
            original = deepcopy(data)
            if reference_plan === nothing
                reference_plan = lower_rkppl(model, data; mod=@__MODULE__, conditioned=(:values,))
            end
            bound = bind_data(reference_plan, data)
            built = build_kernel(bound)
            @test built.layout.total == n
            u = [0.12 * e - 0.3 for e in 1:n]
            @test constrain(built.layout, u).member.offset == u
            @test coordinate_names(built.layout) == [Symbol("member.offset.$e") for e in 1:n]
            expected_fits = nested_locations ?
                [shift[eachindex(y)] .+ off for (y, off) in zip(values, u)] : u .+ shift
            expected_pw = [normal_logpdf.(y, mu) for (y, mu) in zip(values, expected_fits)]
            expected_ll = sum(sum_cell, expected_pw; init=0.0)
            expected_prior = sum(x -> normal_logpdf(x, 0.0), u; init=0.0)
            query(w) = Base.invokelatest(prepare_query(built, bound, w), u)
            @test query((:fitted,)) == expected_fits
            pointwise = query(:pointwise).values
            @test length(pointwise) == n
            @test all(p ≈ q for (p, q) in zip(pointwise, expected_pw))
            @test query(:likelihood) ≈ expected_ll
            @test query(:prior) ≈ expected_prior
            @test query(:sampler) ≈ expected_ll + expected_prior
            sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
            value, gradient = Base.invokelatest(sampler_value_and_gradient!, sampler, similar(u), u)
            @test value ≈ expected_ll + expected_prior
            scores = [sum_cell(y .- mu) for (y, mu) in zip(values, expected_fits)]
            @test gradient ≈ scores .- u
            @test data == original
            push!(programs, Base.remove_linenums!(kernel_expr(bound, built.layout)))
            push!(structures, [(r.kind, r.depth) for r in recipe_inventory(built.spec)])
            source = Base.remove_linenums!(deepcopy(code_expr(sampler.kernel)))
            push!(emitted, (nodes(source), loops(source)))
        end
        # Changing the number and lengths of cells retains iteration in the
        # lowered and prepared programs, including their reachable plate bodies.
        @test all(p -> nodes(p) <= nodes(first(programs)), programs)
        @test all(s -> length(s) <= length(first(structures)), structures)
        @test all(e -> first(e) <= first(first(emitted)), emitted)
        @test all(e -> last(e) > 0, emitted[1:2])
    end
end

@rkppl redefined(y) = begin
    y = 1.0
    return y
end

@testset "Repeated calls retain the single-assignment check" begin
    # Arguments denote inputs. Observing an input is not a redefinition;
    # assigning to that same formal remains the existing single-assignment error.
    data = (; values=[0.1, 0.4])
    model = quote
        @plate for e in eachindex(values)
            member[e] ~ redefined(values[e])
        end
    end
    @test_throws SurfaceLoweringError lower_rkppl(model, data;
        mod=@__MODULE__, conditioned=(:values,))
end

end
