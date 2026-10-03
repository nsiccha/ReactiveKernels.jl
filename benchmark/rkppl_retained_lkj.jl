# Synthetic LKJ preparation and hot-evaluation measurements. Run in a consumer
# environment containing ReactiveKernelsPPL, Enzyme, and DifferentiationInterface.
using ReactiveKernelsPPL, ReactiveKernels, Enzyme, DifferentiationInterface

function lkj_case(K; eta = 2.0)
    # The opening compiler's surface rejected a shape-only matrix as also
    # observation-aligned. Its transform can still be measured on exactly the
    # same bound plan, constructed from an admitted literal declaration.
    baseline = get(ENV, "LKJ_PLAN_BASELINE", "0") == "1"
    dimension = baseline ? K : :(size(M, 2))
    data = Dict(:M => zeros(3, K), :y => [0.1, -0.2, 0.3],
        :x => [0.3, -0.5, 0.7])
    ast = quote
        L ~ LKJCholesky($dimension, $eta)
        mu = L[$K, 1] .* x
        y .~ Normal.(mu, 0.7)
    end
    plan = lower_rkppl(ast, keys(data); conditioned = (:y,))
    if baseline
        p = only(plan.array_parameters)
        dims = Any[:(size(M, 2)), :(size(M, 2))]
        plan = ReactiveKernelsPPL._with(plan; array_parameters = [
            ArrayParameter(p.name, p.family, p.args, dims, p.support_override, p.label)])
    end
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    (; bound, built)
end

function lkj_hot(k::F, q, u; repeats = 1000) where {F}
    g = similar(u)
    k(u)
    sampler_value_and_gradient!(q, g, u)
    value_bytes = @allocated k(u)
    gradient_bytes = @allocated sampler_value_and_gradient!(q, g, u)
    density = minimum(@elapsed(for _ in 1:repeats; k(u); end) for _ in 1:5) / repeats
    gradient = minimum(@elapsed(for _ in 1:repeats; sampler_value_and_gradient!(q, g, u); end)
                       for _ in 1:5) / repeats
    (; density, gradient, value_bytes, gradient_bytes)
end

function lkj_measure(K)
    build = @timed lkj_case(K)
    fx = build.value
    u = [0.2 * sin(i) for i in 1:fx.built.layout.total]
    query = @timed prepare_query(fx.built, fx.bound, :sampler)
    k = query.value
    sampler = @timed prepare_sampler(fx.built, fx.bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    q = sampler.value
    first_density_seconds = @elapsed Base.invokelatest(k, u)
    first_gradient_seconds = @elapsed sampler_value_and_gradient!(q, similar(u), u)
    hot = Base.invokelatest(lkj_hot, k, q, u)
    println((; K, coordinates = length(u), recipes = length(fx.built.spec.graph.recipes),
        build_seconds = build.time, build_bytes = build.bytes,
        prepare_seconds = query.time, prepare_bytes = query.bytes,
        ad_seconds = sampler.time, ad_bytes = sampler.bytes,
        first_density_seconds, first_gradient_seconds, hot...))
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("LKJ benchmark Julia=", VERSION)
    # Warm the common lowering/prepare machinery before measuring dimensions.
    lkj_measure(2)
    for K in (2, 4, 8, 16)
        lkj_measure(K)
    end
end
