# Executable ports of the phase-point and integrator examples on
# ReactiveHMC.jl `main` at ca9ea4ca41924bb0e1fadc01c717e1333916aba6.
module ReactiveHMCExamples

using LinearAlgebra
using ReactiveKernels

export euclidean_examples, riemannian_examples, softabs_examples
export leapfrog!, generalized_leapfrog!, implicit_midpoint!, multistep!, run

struct Counter{F}
    count::Base.RefValue{Int}
    f::F
end

Counter(f) = Counter(Ref(0), f)

function (counter::Counter)(args...; kwargs...)
    counter.count[] += 1
    counter.f(args...; kwargs...)
end

function _reset!(counters)
    for counter in values(counters)
        counter.count[] = 0
    end
    counters
end

_counts(counters) = map(counter -> counter.count[], counters)

function _oracles()
    pot_f(pos) = 0.5 * sum(abs2, pos)
    grad_f(pos) = (pot_f(pos), copy(pos))
    metric_f(pos) = (grad_f(pos)..., Diagonal(1 .+ abs2.(pos)))
    function metric_grad_f(pos)
        pot, dpot, metric = metric_f(pos)
        metric_grad = zeros(eltype(pos), length(pos), length(pos), length(pos))
        for i in eachindex(pos)
            metric_grad[i, i, i] = 2pos[i]
        end
        (pot, dpot, metric, metric_grad)
    end

    (;
        pot = Counter(pot_f),
        grad = Counter(grad_f),
        metric = Counter(metric_f),
        metric_grad = Counter(metric_grad_f),
    )
end

_tr_prod(a::AbstractMatrix, b::AbstractMatrix) = sum(a' .* b)

_metric_solve(metric::Diagonal, rhs) = rhs ./ metric.diag
_metric_solve(metric, rhs) = cholesky(metric) \ rhs
_metric_logdet(metric::Diagonal) = sum(log, metric.diag)
_metric_logdet(metric) = logdet(cholesky(metric))
_metric_inverse(metric::Diagonal) = Diagonal(one.(metric.diag) ./ metric.diag)
_metric_inverse(metric) = Symmetric(inv(cholesky(metric)))

function _metric_dmom(::Val{:gaussian}, metric, mom, speed, mass)
    _metric_solve(metric, mom)
end

function _metric_dmom(::Val{:relativistic}, metric, mom, speed, mass)
    dprekin = _metric_solve(metric, mom)
    sqrt_term = mass * sqrt(1 + dot(mom, dprekin) / (mass * speed)^2)
    dprekin ./ sqrt_term
end

function _metric_hamiltonian(::Val{:gaussian}, pot, metric, mom, speed, mass)
    dkin = _metric_solve(metric, mom)
    pot + 0.5 * (_metric_logdet(metric) + dot(mom, dkin))
end

function _metric_hamiltonian(::Val{:relativistic}, pot, metric, mom, speed, mass)
    dprekin = _metric_solve(metric, mom)
    sqrt_term = mass * sqrt(1 + dot(mom, dprekin) / (mass * speed)^2)
    pot + 0.5 * _metric_logdet(metric) + speed^2 * sqrt_term
end

function _riemannian_dpos(::Val{:gaussian}, mom, metric, inv_metric,
                          metric_grad, dpot, speed, mass)
    dkin = _metric_solve(metric, mom)
    dkin_dpos = map(eachslice(metric_grad; dims = 3)) do partial_metric
        0.5 * _tr_prod(inv_metric, partial_metric) -
            0.5 * dot(dkin, partial_metric, dkin)
    end
    dkin_dpos + dpot
end

function _riemannian_dpos(::Val{:relativistic}, mom, metric, inv_metric,
                          metric_grad, dpot, speed, mass)
    dprekin = _metric_solve(metric, mom)
    sqrt_term = mass * sqrt(1 + dot(mom, dprekin) / (mass * speed)^2)
    dkin = dprekin ./ sqrt_term
    dkin_dpos = map(eachslice(metric_grad; dims = 3)) do partial_metric
        0.5 * _tr_prod(inv_metric, partial_metric) -
            0.5 * dot(dprekin, partial_metric, dkin)
    end
    dkin_dpos + dpot
end

function euclidean_phasepoint_kernels(kinetic::Val, pot_f, grad_f, metric0,
                                      pos0, mom0; speed = 1.0, mass = 1.0)
    pot0 = pot_f(pos0)
    grad0 = grad_f(pos0)
    _, dpot0 = grad0
    chol0 = cholesky(metric0)
    dmom0 = _metric_dmom(kinetic, metric0, mom0, speed, mass)
    ham0 = _metric_hamiltonian(kinetic, pot0, metric0, mom0, speed, mass)

    @kernel spec(pot_f, grad_f, pos::typeof(pos0), mom::typeof(mom0),
                 metric::typeof(metric0)) = begin
        pot::typeof(pot0) = pot_f(pos)
        grad_values::typeof(grad0) = grad_f(pos)
        pot = grad_values[1]
        dpot::typeof(dpot0) = grad_values[2]
        chol::typeof(chol0) = cholesky(metric)
        dham_dpos::typeof(dpot0) = dpot
        dham_dmom::typeof(dmom0) =
            _metric_dmom(kinetic, metric, mom, speed, mass)
        ham::typeof(ham0) =
            _metric_hamiltonian(kinetic, pot, metric, mom, speed, mass)
        return ham
    end

    dpos_kernel = prepare(
        spec; have = (:grad_f, :pos), want = :dham_dpos,
    )
    dmom_kernel = prepare(spec; have = (:metric, :mom), want = :dham_dmom)
    ham_kernel = prepare(
        spec;
        have = (:pot_f, :grad_f, :pos, :mom, :metric),
        want = :ham,
    )

    (;
        spec,
        graph = spec.graph,
        sources = (; pot_f, grad_f, pos = pos0, mom = mom0, metric = metric0),
        prepared = (; dpos = dpos_kernel, dmom = dmom_kernel, ham = ham_kernel),
        geometry = (p -> (; pos = p, metric = metric0)),
        dham_dpos = ((geo, p) -> dpos_kernel(grad_f, geo.pos)),
        dham_dmom = ((geo, p) -> dmom_kernel(geo.metric, p)),
        hamiltonian = ((geo, p) ->
            ham_kernel(pot_f, grad_f, geo.pos, p, geo.metric)),
    )
end

function riemannian_phasepoint_kernels(kinetic::Val, pot_f, grad_f, metric_f,
                                       metric_grad_f, pos0, mom0;
                                       speed = 1.0, mass = 1.0)
    pot0 = pot_f(pos0)
    grad0 = grad_f(pos0)
    _, dpot0 = grad0
    metric_values0 = metric_f(pos0)
    _, _, metric0 = metric_values0
    metric_grad_values0 = metric_grad_f(pos0)
    _, _, _, metric_grad0 = metric_grad_values0
    chol0 = cholesky(metric0)
    inv_metric0 = _metric_inverse(metric0)
    dmom0 = _metric_dmom(kinetic, metric0, mom0, speed, mass)
    dpos0 = _riemannian_dpos(
        kinetic, mom0, metric0, inv_metric0, metric_grad0, dpot0, speed, mass,
    )
    ham0 = _metric_hamiltonian(kinetic, pot0, metric0, mom0, speed, mass)

    @kernel spec(pot_f, grad_f, metric_f, metric_grad_f, metric_inverse_f,
                 pos::typeof(pos0), mom::typeof(mom0)) = begin
        pot::typeof(pot0) = pot_f(pos)
        grad_values::typeof(grad0) = grad_f(pos)
        pot = grad_values[1]
        dpot::typeof(dpot0) = grad_values[2]
        metric_values::typeof(metric_values0) = metric_f(pos)
        pot = metric_values[1]
        dpot = metric_values[2]
        metric::typeof(metric0) = metric_values[3]
        metric_grad_values::typeof(metric_grad_values0) = metric_grad_f(pos)
        pot = metric_grad_values[1]
        dpot = metric_grad_values[2]
        metric = metric_grad_values[3]
        metric_grad::typeof(metric_grad0) = metric_grad_values[4]
        chol::typeof(chol0) = cholesky(metric)
        inv_metric::typeof(inv_metric0) = metric_inverse_f(metric)
        dham_dmom::typeof(dmom0) =
            _metric_dmom(kinetic, metric, mom, speed, mass)
        dham_dpos::typeof(dpos0) = _riemannian_dpos(
            kinetic, mom, metric, inv_metric, metric_grad, dpot, speed, mass,
        )
        ham::typeof(ham0) =
            _metric_hamiltonian(kinetic, pot, metric, mom, speed, mass)
        return ham
    end

    geometry_kernel = prepare(
        spec;
        have = (
            :pot_f, :grad_f, :metric_f, :metric_grad_f,
            :metric_inverse_f, :pos,
        ),
        want = (:pot, :dpot, :metric, :metric_grad, :chol, :inv_metric),
    )
    dpos_kernel = prepare(
        spec;
        have = (:mom, :metric, :inv_metric, :metric_grad, :dpot),
        want = :dham_dpos,
    )
    dmom_kernel = prepare(spec; have = (:metric, :mom), want = :dham_dmom)
    ham_kernel = prepare(spec; have = (:pot, :metric, :mom), want = :ham)

    geometry(p) = begin
        pot, dpot, metric, metric_grad = metric_grad_f(p)
        (;
            pot, dpot, metric, metric_grad,
            chol = cholesky(metric), inv_metric = _metric_inverse(metric),
        )
    end

    (;
        spec,
        graph = spec.graph,
        sources = (;
            pot_f, grad_f, metric_f, metric_grad_f,
            metric_inverse_f = _metric_inverse, pos = pos0, mom = mom0,
        ),
        values = (;
            pos = spec.pos, mom = spec.mom, pot = spec.pot, dpot = spec.dpot,
            metric = spec.metric, metric_grad = spec.metric_grad,
            chol = spec.chol, inv_metric = spec.inv_metric,
            dham_dpos = spec.dham_dpos, dham_dmom = spec.dham_dmom,
            ham = spec.ham,
        ),
        prepared = (;
            geometry = geometry_kernel,
            dpos = dpos_kernel,
            dmom = dmom_kernel,
            ham = ham_kernel,
        ),
        geometry,
        dham_dpos = ((geo, p) -> dpos_kernel(
            p, geo.metric, geo.inv_metric, geo.metric_grad, geo.dpot,
        )),
        dham_dmom = ((geo, p) -> dmom_kernel(geo.metric, p)),
        hamiltonian = ((geo, p) -> ham_kernel(geo.pot, geo.metric, p)),
    )
end

function _softabs_jacobian(eigenvalues, metric_eigenvalues, alpha)
    pre_i = reshape(eigenvalues, :, 1)
    pre_j = reshape(eigenvalues, 1, :)
    metric_i = reshape(metric_eigenvalues, :, 1)
    metric_j = reshape(metric_eigenvalues, 1, :)
    derivative = coth.(alpha .* pre_i) .-
        pre_i .* alpha .* csch.(pre_i .* alpha) .^ 2
    divided_difference = (metric_i .- metric_j) ./ (pre_i .- pre_j)
    ifelse.(pre_i .== pre_j, derivative, divided_difference)
end

function _softabs_geometry(premetric::Diagonal, alpha)
    eigenvalues = premetric.diag
    eigenvectors = Diagonal(one.(eigenvalues))
    metric_eigenvalues = eigenvalues .* coth.(alpha .* eigenvalues)
    q_inv = Diagonal(one.(metric_eigenvalues) ./ metric_eigenvalues)
    jacobian = _softabs_jacobian(eigenvalues, metric_eigenvalues, alpha)
    (; eigenvalues, eigenvectors, metric_eigenvalues, q_inv, jacobian)
end

function _softabs_geometry(premetric, alpha)
    eig = eigen(Symmetric(premetric))
    eigenvalues = eig.values
    eigenvectors = eig.vectors
    metric_eigenvalues = eigenvalues .* coth.(alpha .* eigenvalues)
    q_inv = eigenvectors ./ metric_eigenvalues'
    jacobian = _softabs_jacobian(eigenvalues, metric_eigenvalues, alpha)
    (; eigenvalues, eigenvectors, metric_eigenvalues, q_inv,
       jacobian)
end

function _softabs_dmom(::Val{:gaussian}, geo, mom, speed, mass)
    geo.q_inv * (geo.eigenvectors' * mom)
end

function _softabs_dmom(::Val{:relativistic}, geo, mom, speed, mass)
    dprekin = geo.q_inv * (geo.eigenvectors' * mom)
    sqrt_term = mass * sqrt(1 + dot(mom, dprekin) / (mass * speed)^2)
    dprekin ./ sqrt_term
end

function _softabs_hamiltonian(::Val{:gaussian}, geo, mom, speed, mass)
    dkin = _softabs_dmom(Val(:gaussian), geo, mom, speed, mass)
    geo.pot + 0.5 * (sum(log, geo.metric_eigenvalues) + dot(mom, dkin))
end

function _softabs_hamiltonian(::Val{:relativistic}, geo, mom, speed, mass)
    dprekin = geo.q_inv * (geo.eigenvectors' * mom)
    sqrt_term = mass * sqrt(1 + dot(mom, dprekin) / (mass * speed)^2)
    geo.pot + 0.5 * sum(log, geo.metric_eigenvalues) + speed^2 * sqrt_term
end

function _softabs_dpos(::Val{:gaussian}, geo, mom, speed, mass)
    qpmom = geo.eigenvectors' * mom
    q_inv_qpmom = geo.q_inv * Diagonal(qpmom)
    logdet_term = geo.q_inv * Diagonal(geo.jacobian) * geo.eigenvectors'
    quadratic_term = q_inv_qpmom * geo.jacobian * q_inv_qpmom'
    dkin = map(eachslice(geo.premetric_grad; dims = 3)) do partial_metric
        0.5 * _tr_prod(logdet_term, partial_metric) -
            0.5 * _tr_prod(quadratic_term, partial_metric)
    end
    dkin + geo.dpot
end

function _softabs_dpos(::Val{:relativistic}, geo, mom, speed, mass)
    transformed_mom = geo.eigenvectors' * mom
    dprekin = geo.q_inv * transformed_mom
    sqrt_term = mass * sqrt(1 + dot(mom, dprekin) / (mass * speed)^2)
    logdet_term = geo.q_inv * Diagonal(geo.jacobian) * geo.eigenvectors'
    quadratic_term = geo.q_inv *
        (transformed_mom .* geo.jacobian .* transformed_mom') * geo.q_inv'
    dkin = map(eachslice(geo.premetric_grad; dims = 3)) do partial_metric
        0.5 * _tr_prod(logdet_term, partial_metric) -
            0.5 / sqrt_term * _tr_prod(quadratic_term, partial_metric)
    end
    dkin + geo.dpot
end

function softabs_phasepoint_kernels(kinetic::Val, pot_f, grad_f, premetric_f,
                                    premetric_grad_f, pos0, mom0;
                                    alpha = 20.0, speed = 1.0, mass = 1.0)
    pot0 = pot_f(pos0)
    grad0 = grad_f(pos0)
    _, dpot0 = grad0
    premetric_values0 = premetric_f(pos0)
    _, _, premetric0 = premetric_values0
    premetric_grad_values0 = premetric_grad_f(pos0)
    _, _, _, premetric_grad0 = premetric_grad_values0
    softabs0 = _softabs_geometry(premetric0, alpha)
    geometry0 = (;
        pot = pot0, dpot = dpot0, premetric = premetric0,
        premetric_grad = premetric_grad0,
        eigenvalues = softabs0.eigenvalues,
        eigenvectors = softabs0.eigenvectors,
        metric_eigenvalues = softabs0.metric_eigenvalues,
        q_inv = softabs0.q_inv, jacobian = softabs0.jacobian,
    )
    dpos0 = _softabs_dpos(kinetic, geometry0, mom0, speed, mass)
    dmom0 = _softabs_dmom(kinetic, geometry0, mom0, speed, mass)
    ham0 = _softabs_hamiltonian(kinetic, geometry0, mom0, speed, mass)

    @kernel spec(pot_f, grad_f, premetric_f, premetric_grad_f,
                 softabs_geometry_f, pos::typeof(pos0), mom::typeof(mom0)) = begin
        pot::typeof(pot0) = pot_f(pos)
        grad_values::typeof(grad0) = grad_f(pos)
        pot = grad_values[1]
        dpot::typeof(dpot0) = grad_values[2]
        premetric_values::typeof(premetric_values0) = premetric_f(pos)
        pot = premetric_values[1]
        dpot = premetric_values[2]
        premetric::typeof(premetric0) = premetric_values[3]
        premetric_grad_values::typeof(premetric_grad_values0) =
            premetric_grad_f(pos)
        pot = premetric_grad_values[1]
        dpot = premetric_grad_values[2]
        premetric = premetric_grad_values[3]
        premetric_grad::typeof(premetric_grad0) = premetric_grad_values[4]
        softabs_values::typeof(softabs0) = softabs_geometry_f(premetric, alpha)
        eigenvalues::typeof(softabs0.eigenvalues) = softabs_values.eigenvalues
        eigenvectors::typeof(softabs0.eigenvectors) = softabs_values.eigenvectors
        metric_eigenvalues::typeof(softabs0.metric_eigenvalues) =
            softabs_values.metric_eigenvalues
        q_inv::typeof(softabs0.q_inv) = softabs_values.q_inv
        jacobian::typeof(softabs0.jacobian) = softabs_values.jacobian
        dham_dpos::typeof(dpos0) = _softabs_dpos(
            kinetic,
            (;
                pot, dpot, premetric, premetric_grad, eigenvalues,
                eigenvectors, metric_eigenvalues, q_inv, jacobian,
            ),
            mom, speed, mass,
        )
        dham_dmom::typeof(dmom0) = _softabs_dmom(
            kinetic,
            (;
                pot, dpot, premetric, premetric_grad, eigenvalues,
                eigenvectors, metric_eigenvalues, q_inv, jacobian,
            ),
            mom, speed, mass,
        )
        ham::typeof(ham0) = _softabs_hamiltonian(
            kinetic,
            (;
                pot, dpot, premetric, premetric_grad, eigenvalues,
                eigenvectors, metric_eigenvalues, q_inv, jacobian,
            ),
            mom, speed, mass,
        )
        return ham
    end

    geometry_kernel = prepare(
        spec;
        have = (
            :pot_f, :grad_f, :premetric_f, :premetric_grad_f,
            :softabs_geometry_f, :pos,
        ),
        want = (
            :pot, :dpot, :premetric, :premetric_grad, :eigenvalues,
            :eigenvectors, :metric_eigenvalues, :q_inv, :jacobian,
        ),
    )

    geometry(p) = begin
        pot, dpot, premetric, premetric_grad = premetric_grad_f(p)
        softabs_values = _softabs_geometry(premetric, alpha)
        (;
            pot, dpot, premetric, premetric_grad,
            eigenvalues = softabs_values.eigenvalues,
            eigenvectors = softabs_values.eigenvectors,
            metric_eigenvalues = softabs_values.metric_eigenvalues,
            q_inv = softabs_values.q_inv,
            jacobian = softabs_values.jacobian,
        )
    end

    (;
        spec,
        graph = spec.graph,
        sources = (;
            pot_f, grad_f, premetric_f, premetric_grad_f,
            softabs_geometry_f = _softabs_geometry, pos = pos0, mom = mom0,
        ),
        prepared = (; geometry = geometry_kernel),
        geometry,
        dham_dpos = ((geo, p) -> _softabs_dpos(kinetic, geo, p, speed, mass)),
        dham_dmom = ((geo, p) -> _softabs_dmom(kinetic, geo, p, speed, mass)),
        hamiltonian = ((geo, p) -> _softabs_hamiltonian(
            kinetic, geo, p, speed, mass,
        )),
    )
end

function leapfrog!(pos, mom, kernels; stepsize)
    geometry = kernels.geometry(pos)
    mom .-= 0.5 * stepsize .* kernels.dham_dpos(geometry, mom)
    pos .+= stepsize .* kernels.dham_dmom(geometry, mom)
    geometry = kernels.geometry(pos)
    mom .-= 0.5 * stepsize .* kernels.dham_dpos(geometry, mom)
    (; pos, mom, ham = kernels.hamiltonian(geometry, mom), geometry)
end

function implicit_midpoint!(pos, mom, kernels; stepsize, n_fi_steps)
    pos0, mom0 = copy(pos), copy(mom)
    geometry = kernels.geometry(pos)

    for _ in 1:n_fi_steps
        dham_dmom = kernels.dham_dmom(geometry, mom)
        dham_dpos = kernels.dham_dpos(geometry, mom)
        @. pos = pos0 + 0.5 * stepsize * dham_dmom
        @. mom = mom0 - 0.5 * stepsize * dham_dpos
        geometry = kernels.geometry(pos)
    end

    @. pos = 2pos - pos0
    @. mom = 2mom - mom0
    geometry = kernels.geometry(pos)
    (; pos, mom, ham = kernels.hamiltonian(geometry, mom), geometry)
end


function multistep!(integrator!, pos, mom, kernels; n_steps, stepsize, kwargs...)
    result = nothing
    substepsize = stepsize / n_steps
    for _ in 1:n_steps
        result = integrator!(pos, mom, kernels; stepsize = substepsize, kwargs...)
    end
    result
end

function generalized_leapfrog!(pos, mom, kernels; stepsize, n_fi_steps)
    pos0, mom0 = copy(pos), copy(mom)
    geometry = kernels.geometry(pos)

    for _ in 1:n_fi_steps
        dham_dpos = kernels.dham_dpos(geometry, mom)
        @. mom = mom0 - 0.5 * stepsize * dham_dpos
    end

    dham_dmom0 = copy(kernels.dham_dmom(geometry, mom))
    for _ in 1:n_fi_steps
        dham_dmom = kernels.dham_dmom(geometry, mom)
        @. pos = pos0 + 0.5 * stepsize * (dham_dmom0 + dham_dmom)
        geometry = kernels.geometry(pos)
    end

    dham_dpos = kernels.dham_dpos(geometry, mom)
    @. mom -= 0.5 * stepsize * dham_dpos
    (; pos, mom, ham = kernels.hamiltonian(geometry, mom), geometry)
end

function euclidean_examples()
    pos0 = [0.25, -0.5]
    mom0 = [0.4, 0.1]
    metric = Diagonal(ones(2))

    gaussian_oracles = _oracles()
    gaussian = euclidean_phasepoint_kernels(
        Val(:gaussian), gaussian_oracles.pot, gaussian_oracles.grad,
        metric, pos0, mom0,
    )
    _reset!(gaussian_oracles)
    gaussian_result = leapfrog!(copy(pos0), copy(mom0), gaussian; stepsize = 0.1)

    relativistic_oracles = _oracles()
    relativistic = euclidean_phasepoint_kernels(
        Val(:relativistic), relativistic_oracles.pot, relativistic_oracles.grad,
        metric, pos0, mom0; speed = 1.5, mass = 0.8,
    )
    _reset!(relativistic_oracles)
    relativistic_result = generalized_leapfrog!(
        copy(pos0), copy(mom0), relativistic; stepsize = 0.1, n_fi_steps = 3,
    )

    (;
        euclidean_phasepoint = gaussian_result,
        relativistic_euclidean_phasepoint = relativistic_result,
        gaussian_calls = _counts(gaussian_oracles),
        relativistic_calls = _counts(relativistic_oracles),
        gaussian,
        relativistic,
    )
end

function _phasepoint(kernels, pos, mom)
    geometry = kernels.geometry(pos)
    (;
        dham_dpos = kernels.dham_dpos(geometry, mom),
        dham_dmom = kernels.dham_dmom(geometry, mom),
        ham = kernels.hamiltonian(geometry, mom),
        geometry,
    )
end

function riemannian_examples()
    pos0 = [0.25, -0.5]
    mom0 = [0.4, 0.1]

    gaussian_oracles = _oracles()
    gaussian = riemannian_phasepoint_kernels(
        Val(:gaussian), gaussian_oracles.pot, gaussian_oracles.grad,
        gaussian_oracles.metric, gaussian_oracles.metric_grad, pos0, mom0,
    )
    _reset!(gaussian_oracles)
    initial = _phasepoint(gaussian, pos0, mom0)
    _reset!(gaussian_oracles)
    integrated = generalized_leapfrog!(
        copy(pos0), copy(mom0), gaussian; stepsize = 0.1, n_fi_steps = 4,
    )
    generalized_calls = _counts(gaussian_oracles)
    _reset!(gaussian_oracles)
    midpoint = implicit_midpoint!(
        copy(pos0), copy(mom0), gaussian; stepsize = 0.1, n_fi_steps = 4,
    )

    relativistic_oracles = _oracles()
    relativistic = riemannian_phasepoint_kernels(
        Val(:relativistic), relativistic_oracles.pot, relativistic_oracles.grad,
        relativistic_oracles.metric, relativistic_oracles.metric_grad, pos0, mom0;
        speed = 1.5, mass = 0.8,
    )
    _reset!(relativistic_oracles)
    relativistic_result = _phasepoint(relativistic, pos0, mom0)

    (;
        riemannian_phasepoint = initial,
        relativistic_riemannian_phasepoint = relativistic_result,
        generalized_leapfrog = integrated,
        generalized_leapfrog_calls = generalized_calls,
        implicit_midpoint = midpoint,
        implicit_midpoint_calls = _counts(gaussian_oracles),
        relativistic_calls = _counts(relativistic_oracles),
        gaussian,
        relativistic,
    )
end

function softabs_examples()
    pos0 = [0.25, -0.5]
    mom0 = [0.4, 0.1]

    gaussian_oracles = _oracles()
    gaussian = softabs_phasepoint_kernels(
        Val(:gaussian), gaussian_oracles.pot, gaussian_oracles.grad,
        gaussian_oracles.metric, gaussian_oracles.metric_grad, pos0, mom0,
    )
    _reset!(gaussian_oracles)
    gaussian_result = _phasepoint(gaussian, pos0, mom0)
    _reset!(gaussian_oracles)
    integrated = multistep!(
        generalized_leapfrog!, copy(pos0), copy(mom0), gaussian;
        stepsize = 0.02, n_fi_steps = 2, n_steps = 3,
    )

    relativistic_oracles = _oracles()
    relativistic = softabs_phasepoint_kernels(
        Val(:relativistic), relativistic_oracles.pot, relativistic_oracles.grad,
        relativistic_oracles.metric, relativistic_oracles.metric_grad, pos0, mom0;
        speed = 1.5, mass = 0.8,
    )
    _reset!(relativistic_oracles)
    relativistic_result = _phasepoint(relativistic, pos0, mom0)

    (;
        riemannian_softabs_phasepoint = gaussian_result,
        relativistic_riemannian_softabs_phasepoint = relativistic_result,
        generalized_multistep = integrated,
        generalized_multistep_calls = _counts(gaussian_oracles),
        relativistic_calls = _counts(relativistic_oracles),
        gaussian,
        relativistic,
    )
end

function run(io::IO = stdout)
    euclidean = euclidean_examples()
    riemannian = riemannian_examples()
    softabs = softabs_examples()

    println(io, "ReactiveHMC.jl compatibility examples")
    println(io, "  euclidean_phasepoint ham: ", euclidean.euclidean_phasepoint.ham)
    println(io, "  relativistic_euclidean_phasepoint ham: ",
            euclidean.relativistic_euclidean_phasepoint.ham)
    println(io, "  riemannian_phasepoint ham: ", riemannian.riemannian_phasepoint.ham)
    println(io, "  relativistic_riemannian_phasepoint ham: ",
            riemannian.relativistic_riemannian_phasepoint.ham)
    println(io, "  riemannian_softabs_phasepoint ham: ",
            softabs.riemannian_softabs_phasepoint.ham)
    println(io, "  relativistic_riemannian_softabs_phasepoint ham: ",
            softabs.relativistic_riemannian_softabs_phasepoint.ham)
    println(io, "  generalized-leapfrog oracle calls: ",
            riemannian.generalized_leapfrog_calls)
    nothing
end

end # module ReactiveHMCExamples

if abspath(PROGRAM_FILE) == @__FILE__
    ReactiveHMCExamples.run()
end
