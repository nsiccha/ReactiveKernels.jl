using Reactant

function _event_test_compiled(plan, built)
    u = [0.1cos(i) for i in 1:built.layout.total]
    names = sort!(collect(keys(plan.columns)))
    bound = NamedTuple{Tuple(names)}(Tuple(plan.columns[n] for n in names))
    curve = Base.invokelatest(prepare, built.spec;
        have=(:unconstrained, names...), bound, want=:curve_sum)
    return Base.invokelatest(_event_test_compile_raw, curve, u)
end

function _event_test_compile_raw(curve, u)
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize=false curve(ru))
    operations = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith)\.\w+", hlo)
        operations[m.match] = get(operations, m.match, 0) + 1
    end
    @test !isempty(operations)
    compiled = Reactant.@compile curve(ru)
    @test Float64(compiled(ru)) ≈ curve(u) rtol=1e-9
    ad = prepare_ad(curve, AutoEnzyme(; mode=Enzyme.Reverse), u; active=:unconstrained)
    value, gradient = ReactiveKernels.ad_value_and_gradient!(ad, similar(u), u)
    cad = compile_ad_value_and_gradient(ad, ru)
    rvalue, rgradient = cad(ru)
    @test Float64(rvalue) ≈ value rtol=1e-9
    @test Array(rgradient) ≈ gradient rtol=1e-9 atol=1e-10
    return operations
end

@testset "event LP: compiled curve retains structure as schedules grow" begin
    structures = Dict{String,Int}[]
    for (G, k) in ((2, 3), (4, 5))
        model = merge(_event_test_ast(; k), :(curve_sum = sum(log_F)))
        plan = model(; _event_test_data(G)...)
        built = build_kernel(plan)
        println("EVENT_CURVE_COMPILED_BEGIN subjects=", G, " k=", k)
        flush(stdout)
        push!(structures, _event_test_compiled(plan, built))
        println("EVENT_CURVE_COMPILED_PASS subjects=", G, " k=", k)
        flush(stdout)
    end
    @test first(structures) == last(structures)
end
