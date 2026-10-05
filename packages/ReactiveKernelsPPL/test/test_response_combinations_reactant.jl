using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Reactant, Test

function _rc_compiled_fixture(label, n)
    if label === :mi || label === :intercept
        jobs = collect(1:2:n)
        y = fill(0.2, length(jobs))
        wt = 1 .+ collect(1:n) ./ n
        cols = Dict{Symbol,AbstractVector}(:y=>y, :Jobs=>jobs, :wt=>wt)
        evidence = label === :mi ? ResponseEvidence(:truncated, -1.0, 1.5) :
            ResponseEvidence(:none, nothing, nothing)
        range = label === :mi ? (3:n) : nothing
        r = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, 0.5,
            label === :mi ? :wt : nothing, evidence, :y_resp, nothing, range;
            mi_jobs=:Jobs)
        plan = StructuralPlan([r],
            [PredictorSpec(:mu, IdentityLink,
                [TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept, :intercept)], :mu)],
            [PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
            SampledParameter[], AssignmentSpec[], cols, n)
        oracle = nt -> begin
            d = label === :mi ? truncated(Normal(nt.mu[1], 0.5), -1.0, 1.5) :
                Normal(nt.mu[1], 0.5)
            rows = label === :mi ? jobs[2:end] : jobs
            ll = sum(j -> (label === :mi ? wt[j] : 1.0) * logpdf(d, 0.2), rows)
            ll + logpdf(Normal(0, 1), nt.mu[1])
        end
        return plan, oracle
    elseif label === :plate
        y = fill(0.2, n)
        wt = 1 .+ collect(1:n) ./ n
        cols = Dict{Symbol,AbstractVector}(:y=>y, :wt=>wt)
        r = LikelihoodSpec(GaussianFam, IdentityLink, :y, :z, 0.5, :wt,
            ResponseEvidence(:truncated, -1.0, 1.5), :y_resp, nothing, 1:n)
        plan = StructuralPlan([r], PredictorSpec[], PopulationPrior[],
            SampledParameter[], AssignmentSpec[], cols, n;
            plate_parameters=[PlateParameter(:z, :normal, (arg1=0.0,arg2=1.0), nothing)])
        oracle = nt -> sum(wt .* map(i -> logpdf(
            truncated(Normal(nt.z[i], 0.5), -1.0, 1.5), y[i]), eachindex(y))) +
            sum(logpdf.(Normal(0, 1), nt.z))
        return plan, oracle
    end
    if label === :trials
        ast = quote
            p ~ Beta(2, 2)
            y .~ MixtureModel.(vcat.(Binomial.(n1, p), Binomial.(n2, 0.3)),
                Ref([0.3, 0.7]))
        end
        data = (; y=fill(3,n), n1=fill(1,n), n2=fill(4,n))
        oracle = nt -> n*logpdf(MixtureModel([Binomial(1,nt.p),Binomial(4,0.3)],
            [0.3,0.7]),3) + logpdf(Beta(2,2),nt.p) + log(nt.p)+log1p(-nt.p)
    elseif label === :range
        # Full coverage, as required by USER 1uhcm3b even for missing entries.
        ast = quote
            m ~ Normal(0, 1)
            y[1:$n] .~ MixtureModel.(vcat.(Normal.(m, 0.5), Normal.(1.0, 0.5)),
                Ref([0.4, 0.6]))
        end
        data = (; y=fill(0.2,n))
        oracle = nt -> n*logpdf(MixtureModel([Normal(nt.m,0.5),Normal(1,0.5)],
            [0.4,0.6]),0.2) + logpdf(Normal(0,1),nt.m)
    elseif label === :censored
        ast = quote
            m ~ Normal(0, 1)
            y .~ weighted.(censored.(MixtureModel.(vcat.(Normal.(m, 0.5),
                Normal.(1.0, 0.5)), Ref([0.4, 0.6])), -0.5, 1.5), wt)
        end
        data = (; y=repeat([-0.5,0.2,1.5,0.2], n÷4), wt=1 .+ collect(1:n)./n)
        oracle = nt -> sum(data.wt .* logpdf.(censored(
            MixtureModel([Normal(nt.m,0.5),Normal(1,0.5)],[0.4,0.6]),
            -0.5,1.5),data.y)) + logpdf(Normal(0,1),nt.m)
    else
        ast = quote
            s ~ Exponential(1)
            y .~ weighted.(MixtureModel.(vcat.(Normal.(x, s), Normal.(0.0, s)),
                Ref(w)), wt)
        end
        data = (; y=fill(0.2,n), x=collect(1:n)./n, w=[0.3,0.7], wt=1 .+ collect(1:n)./n)
        oracle = nt -> sum(data.wt .* map(i -> logpdf(MixtureModel(
            [Normal(data.x[i],nt.s),Normal(0,nt.s)],data.w),data.y[i]),1:n)) +
            logpdf(Exponential(1),nt.s) + log(nt.s)
    end
    plan = bind_data(lower_rkppl(ast,data;conditioned=(:y,)),Dict{Symbol,Any}(pairs(data)))
    return plan, oracle
end

@testset "response combinations compile without row body replication" begin
    emitted = Dict{Symbol,Any}()
    # Each censored arm has at least two rows, keeping singleton shapes out
    # of this same-cell growth regression. Backend optimization may expand
    # small batches or choose loops: constraints.md scopes this check to RK
    # emission (USER 1rvu25u / October 5 scope clarification).
    for n in (8, 16), label in (:weights, :trials, :range, :mi, :intercept, :plate, :censored)
        @testset "$label / n=$n" begin
            plan, oracle = _rc_compiled_fixture(label, n)
            saved_columns = deepcopy(plan.columns)
            built = build_kernel(plan)
            u = [0.2sin(i) for i in 1:built.layout.total]
            sampler = Base.invokelatest(prepare_sampler,built,plan,u;
                backend=AutoEnzyme(;mode=Enzyme.Reverse))
            ru = Reactant.to_rarray(u)
            kernel = sampler.kernel
            compiled = Reactant.@compile kernel(ru)
            cad = Base.invokelatest(compile_ad_value_and_gradient,sampler.ad,ru)
            # The negative mean crosses the truncated-normal normalization
            # branch in :mi and :plate, reusing the same compiled executable.
            for probe in (u, u .- 1.6)
                saved_probe = copy(probe)
                reference = oracle(constrain(built.layout,probe))
                grad = similar(probe)
                val, _ = Base.invokelatest(sampler_value_and_gradient!,sampler,grad,probe)
                @test val ≈ reference
                rprobe = Reactant.to_rarray(probe)
                @test Float64(compiled(rprobe)) ≈ reference
                rval, rgrad = cad(rprobe)
                @test Float64(rval) ≈ reference
                @test Array(rgrad) ≈ grad rtol=2e-5 atol=2e-7
                h = cbrt(eps(Float64))
                fd = map(eachindex(probe)) do i
                    up, down = copy(probe), copy(probe)
                    up[i] += h
                    down[i] -= h
                    (oracle(constrain(built.layout,up))-oracle(constrain(built.layout,down)))/(2h)
                end
                @test grad ≈ fd rtol=2e-5 atol=2e-7
                @test Array(rgrad) ≈ fd rtol=2e-5 atol=2e-7
                @test probe == saved_probe
            end
            @test isequal(plan.columns, saved_columns)
            ad = sampler.ad
            both(v) = ad_value_and_gradient(ad, v)
            modules = (Reactant.@code_hlo(optimize=false, kernel(ru)),
                Reactant.@code_hlo(kernel(ru)),
                Reactant.@code_hlo(optimize=false, both(ru)),
                Reactant.@code_hlo(both(ru)))
            structures = _ppl_mlir_structure.(modules)
            raw = (structures[1], structures[3])
            @test all(s -> !isempty(s.cells), raw)
            # Same authored cells and call sites at 8/16 rows; sizes and bound
            # constant payloads may change. Inspect reachable bodies too, so
            # an empty retained batch cannot make this test pass.
            cells = map(s -> s.cells, raw)
            @test all(cell -> "func.return" in cell, Iterators.flatten(cells))
            n == 8 ? (emitted[label]=cells) : (@test cells == emitted[label])
            println("response combination ", label, " n=", n, " inventories=",
                map(structures) do s
                    sort!(collect(Dict(op=>count(==(op),s.operations)
                        for op in unique(s.operations))); by=first)
                end)
            if haskey(ENV, "RKPPL_RESPONSE_COMBINATIONS_HLO_DIR")
                dir = ENV["RKPPL_RESPONSE_COMBINATIONS_HLO_DIR"]
                mkpath(dir)
                for (phase, mod) in zip(("primal-raw", "primal-optimized",
                        "reverse-raw", "reverse-optimized"), modules)
                    write(joinpath(dir, "$label-$n-$phase.mlir"), String(mod))
                end
            end
        end
    end
end
