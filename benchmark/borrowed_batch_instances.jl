# Small generic lifecycle probe, independent of any application benchmark.
# Run: julia --startup-file=no --project=. benchmark/borrowed_batch_instances.jl
using ReactiveKernels

@kernel instance_curve(position, data; amount=1.0) = begin
    units = position .* data
    result = (; curve=units .* amount, summary=(total=sum(units) * amount,))
end

instance_target(kernel::ReactiveKernels._KernelSignatureCallable) = kernel.target
instance_target(kernel) = kernel
instance_factory() = prepare_batched(instance_curve; batched=:position, reuse=true)

function measure_instances(label, factory, count)
    factory() # compile the measurement call before timing
    sample = @timed [factory() for _ in 1:count]
    println(label, " count=", count, " seconds_per_instance=", sample.time / count,
            " bytes_per_instance=", sample.bytes / count)
    sample.value
end

function main()
    template = instance_factory()
    pristine = measure_instances("COPY_PRISTINE", () -> copy(template), 1000)
    measure_instances("DEEPCOPY_PRISTINE", () -> deepcopy(template), 100)
    template(collect(1.0:64.0), collect(1.0:512.0))
    used = measure_instances("COPY_USED", () -> copy(template), 1000)
    measure_instances("DEEPCOPY_USED", () -> deepcopy(template), 100)
    measure_instances("REPREPARE", instance_factory, 10)

    raw = instance_target(template)
    instances = (pristine..., used...)
    @assert all(instance -> instance_target(instance).native === raw.native, instances)
    @assert all(instance -> instance_target(instance).target === raw.target, instances)
    @assert all(instance -> code_expr(instance) === code_expr(template), instances)
    @assert all(instance -> all(slot[] === nothing for slot in
                               instance_target(instance).caches), instances)
    @assert all(instance -> instance_target(instance).caches[1] !== raw.caches[1], instances)
    @assert length(Set(objectid(instance_target(instance).caches[1])
                       for instance in instances)) == length(instances)
    println("INSTANCE_WORK prepared_templates=1 copies=", length(instances),
            " shared_graphs=1 shared_asts=1 shared_callables=1 independent_empty_caches=",
            length(instances))

    # Reuse one instance across the request's sequential amounts. Publish only
    # compact owned values; all borrowed leaves stay inside this operation.
    reader = copy(template)
    summaries = map((0.5, 1.0, 2.0)) do amount
        result = reader([1.0, 2.0], [3.0, 4.0]; amount)
        sum(result.summary.total)
    end
    @assert summaries == (10.5, 21.0, 42.0)
    println("SEQUENTIAL_COMPACT_RESULT ", summaries)
end

main()
