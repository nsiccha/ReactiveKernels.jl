using ReactiveKernels, ReactiveKernelsPPL, Test

# JavaScript modules of an RKPPL prediction query: the query `prepare_query`
# returns for a bound model, emitted on its own and called from a picture's
# kernel, keeps the binding's data and the model's transforms (snag
# emit-an-rk-pictu-b9fa64e8).

const JS_QUERY_NODE = ReactiveKernels._js_default_node() !== nothing

const JS_QUERY_MODEL = @rkppl begin
    e ~ Gamma(1.0, 1.0)
    lp = e .* log.(w)
    p = exp.(lp)
    k .~ Binomial.(n, p)
end
const JS_QUERY_PLAN = JS_QUERY_MODEL(; w = [0.15, 0.35, 0.6, 0.8], n = zeros(Int, 4)) |
                      (; k = zeros(Int, 4))
const JS_QUERY_BUILT = build_kernel(JS_QUERY_PLAN)
const JS_QUERY = prepare_query(JS_QUERY_BUILT, JS_QUERY_PLAN, (:p,))

@kernel js_query_picture(unconstrained) = begin
    probability = JS_QUERY(unconstrained)
    return probability
end

# The last two points are not of this binding's packed length: the module
# raises the query's `ContractValidationError`, as native does.
const JS_QUERY_CASES = [([log(0.25)],), ([0.0],), ([log(4.0)],), ([-800.0],), ([800.0],),
                        (Float64[],), ([0.0, 1.0],)]

function js_query_check(native, source, entry)
    for (c, r) in zip(JS_QUERY_CASES, ReactiveKernels.js_run(source, entry, JS_QUERY_CASES))
        expected = try
            native(c...)
        catch e
            e
        end
        if expected isa Exception
            @test r.julia_error && r.error == string(nameof(typeof(expected)))
        else
            @test r.error === nothing
            r.error === nothing || continue
            @test typeof(r.value) == typeof(expected)
            @test isapprox(r.value, expected; rtol = 1e-12, atol = 1e-300)
        end
    end
end

@testset "js_source of an RKPPL prediction query" begin
    direct = ReactiveKernels.js_source(JS_QUERY; types = (; unconstrained = Vector{Float64}), name = :query)
    picture = ReactiveKernels.js_source(js_query_picture; have = (:unconstrained,), want = :probability,
                                        types = (; unconstrained = Vector{Float64}), name = :picture)
    # the binding's data (here `log.(w)`, which the binding computes) are part
    # of each module, and the packed-length check runs at its entry
    for source in (direct, picture)
        @test count("Object.freeze([", source) == 1
        @test occursin("ppl_check_packed(unconstrained, 1);", source)
    end
    @test occursin("function _rk_prepared_p(unconstrained) {", picture)
    if JS_QUERY_NODE
        js_query_check(JS_QUERY, direct, :query)
        js_query_check(JS_QUERY, picture, :picture)
    else
        @test_skip "no node on the PATH"
    end
end
