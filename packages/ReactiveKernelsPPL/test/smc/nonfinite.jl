# Runnable alone in the optional SMC integration environment, or from runtests.
module SMCNonfiniteTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random, Test, Serialization

const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)

struct Endpoint{F,G}
    density::F
    gradient::G
end
EXT._density(t::Endpoint, u) = t.density(u)
function EXT._gradient(t::Endpoint, g, u)
    value, gradient = t.gradient(u)
    copyto!(g, gradient)
    return value
end
endpoint(value, gradient) = Endpoint(u -> value, u -> (value, gradient))
const FINITE = Endpoint(u -> -sum(abs2, u)/2, u -> (-sum(abs2, u)/2, -u))

function caught(f)
    try
        f()
    catch err
        return err, catch_backtrace()
    end
    error("expected an evaluation failure")
end

function state_for(source, target; temperature=0.0)
    n = 4
    # Keep the first particle at zero for the exact diagnostic assertions,
    # but exercise numerical failures from a cloud that has not collapsed.
    EXT.SMCState(source, target, reshape(collect(0.0:n-1),1,n), fill(1/n,n), temperature, 0.0,
        Xoshiro(741), SMCOptions(moves=1), collect(1:n), collect(1:n), EXT.SMCStage[], 0, 0)
end

@testset "Exact endpoint diagnostics and support semantics" begin
    position, gradient = [0.25,-0.5], [NaN,Inf]
    snapshot = SMCNonfiniteError(position,NaN,gradient,:density,:target,0.4)
    position .= 0
    gradient .= 0
    @test snapshot.position == [0.25,-0.5]
    @test isequal(snapshot.gradient,[NaN,Inf])
    u = [0.25, -0.5]
    for beta in (0.0, 0.4, 1.0), side in (:source, :target), bad in (NaN, Inf)
        a, b = side === :source ? (endpoint(bad, [0.0,0.0]), FINITE) :
            (FINITE, endpoint(bad, [0.0,0.0]))
        bridge = EXT.BridgeTarget(a, b, beta, 2)
        active = side === :source ? beta != 1 : beta != 0
        for evaluation in (LogDensityProblems.logdensity, LogDensityProblems.logdensity_and_gradient)
            if active
                err, bt = caught(() -> evaluation(bridge,u))
                @test err isa SMCNonfiniteError
                @test isequal(err.value, bad)
                @test err.position == u && err.position !== u
                @test err.endpoint === side
                @test err.temperature == beta
                @test err.quantity === :density
                @test ncodeunits(sprint(showerror, err, bt)) < 512
            else
                result = evaluation(bridge,u)
                @test (result isa Tuple ? first(result) : result) == -sum(abs2,u)/2
            end
        end
    end
    for bad in (NaN, Inf, -Inf), side in (:source, :target), beta in (0.0,0.4,1.0)
        g = [bad, 1.0]
        a, b = side === :source ? (endpoint(2.0,g),FINITE) : (FINITE,endpoint(2.0,g))
        bridge = EXT.BridgeTarget(a,b,beta,2)
        active = side === :source ? beta != 1 : beta != 0
        if active
            err, _ = caught(() -> LogDensityProblems.logdensity_and_gradient(bridge,u))
            @test err isa SMCNonfiniteError
            @test err.quantity === :gradient && err.endpoint === side
            @test err.temperature == beta && err.value == 2.0
            @test isequal(err.gradient,g) && err.gradient !== g
            saved = copy(err.gradient)
            g .= 0
            @test isequal(err.gradient,saved)
        else
            @test first(LogDensityProblems.logdensity_and_gradient(bridge,u)) == -sum(abs2,u)/2
        end
    end
    unsupported = endpoint(-Inf, fill(NaN,2))
    for beta in (0.0,0.4,1.0)
        bridge = EXT.BridgeTarget(FINITE,unsupported,beta,2)
        expected = beta == 0 ? -sum(abs2,u)/2 : -Inf
        @test LogDensityProblems.logdensity(bridge,u) == expected
        @test first(LogDensityProblems.logdensity_and_gradient(bridge,u)) == expected
    end
    for (a,b) in ((-1e308,1e308),(1e308,-1e308))
        state = state_for(endpoint(a,[0.0]),endpoint(b,[0.0]))
        err, _ = caught(() -> smc_step(state))
        @test err isa SMCNonfiniteError
        @test err.quantity === :logratio && err.endpoint === :bridge
        @test err.value == b-a && err.position == [0.0]
    end
end

@testset "Mutation task propagation, exact proposals and rollback" begin
    for quantity in (:density,:gradient), bad in (NaN,Inf,-Inf)
        quantity === :density && bad == -Inf && continue
        observed = Ref(Float64[])
        target = Endpoint(FINITE.density, u -> begin
            if any(!iszero,u)
                observed[] = copy(u)
                quantity === :density ? (bad,-u) : (-sum(abs2,u)/2,fill(bad,length(u)))
            else
                FINITE.gradient(u)
            end
        end)
        state = state_for(FINITE,target)
        saved = deepcopy(state)
        err, bt = caught(() -> smc_step(state))
        @test err isa SMCNonfiniteError # not TaskFailedException
        @test err.position == observed[] && any(!iszero,err.position)
        @test err.endpoint === :target && err.temperature == 1.0
        @test err.quantity === quantity
        observed[] .= 999
        @test all(!=(999),err.position)
        @test state.positions == saved.positions && state.weights == saved.weights
        @test state.rng == saved.rng && state.history == saved.history
        @test state.stage == 0 && state.temperature == 0
        @test ncodeunits(sprint(showerror,err,bt)) < 512
        @test ncodeunits(sprint(Base.display_error,err,bt)) < 512
        @test ncodeunits(sprint(showerror,CapturedException(err,bt))) < 512
        @test !occursin("nested task",sprint(showerror,err,bt))
        io = IOBuffer()
        serialize(io,err)
        restored = deserialize(seekstart(io))
        @test isequal(restored.gradient,err.gradient) && restored.position == err.position
    end
    for side in (:source,:target)
        bad = Endpoint(FINITE.density, u -> (0.0,fill(NaN,length(u))))
        state = side === :source ? state_for(bad,FINITE) : state_for(FINITE,bad)
        err, _ = caught(() -> EXT._library_stage(state,0.375,zeros(4),UInt64(9)))
        @test err isa SMCNonfiniteError
        @test err.endpoint === side && err.temperature == 0.375
    end
    # Unrelated callback failures must not become numerical diagnostics.
    target = Endpoint(FINITE.density, u -> throw(ArgumentError("unrelated callback failure")))
    err, _ = caught(() -> smc_step(state_for(FINITE,target)))
    @test err isa TaskFailedException
    @test occursin("unrelated callback failure",sprint(showerror,err))

    # Both the array size and prepared type depth used to inflate error output.
    huge = SMCNonfiniteError(zeros(100_000),NaN,fill(Inf,100_000),:density,:target,0.3)
    @test ncodeunits(sprint(showerror,huge)) < 512
    @test ncodeunits(sprint(show,huge)) < 512
end

function model_target(y)
    src = quote
        mu ~ Normal(0,1)
        y .~ Normal.(mu,1)
    end
    plan = bind_data(lower_rkppl(src,(; y);conditioned=(:y,)),(; y))
    built = build_kernel(plan)
    prepare_smc(built,plan,[0.0];backend=AutoEnzyme(mode=Enzyme.Reverse),
        model_id="nonfinite-fixture",binding_id="same-shape")
end

@testset "Public initial fit, data update and checkpoint diagnostics" begin
    invalid = model_target([NaN])
    start = initialize_smc(invalid;rng=Xoshiro(85),particles=4)
    err, _ = caught(() -> smc_step(start))
    @test err isa SMCNonfiniteError
    @test err.position == start.positions[:,1] && isnan(err.value)
    @test err.temperature == 0.0 && err.endpoint === :target
    @test_throws SMCNonfiniteError smc(invalid;rng=Xoshiro(85),particles=4)
    @test_throws SMCNonfiniteError smc_checkpoint(start)
    valid = model_target([0.0])
    fit = smc(valid;rng=Xoshiro(85),particles=4)
    saved = deepcopy(fit.positions)
    rng = copy(fit.rng)
    err, _ = caught(() -> update_smc(fit,invalid))
    @test err isa SMCNonfiniteError
    @test err.position == fit.positions[:,1]
    @test err.endpoint === :target && err.temperature == 0.0
    @test fit.positions == saved && fit.rng == rng && fit.temperature == 1.0
    cp = smc_checkpoint(initialize_smc(valid;rng=Xoshiro(85),particles=4))
    err, _ = caught(() -> resume_smc(invalid,cp))
    @test err isa SMCNonfiniteError
    @test err.position == cp.positions[:,1] && err.endpoint === :target
end
end
