using ReactiveKernels, Reactant, Test
import Enzyme
Reactant.set_default_backend("cpu")

module SharedHostArrayBranchFixtures
using ReactiveKernels, Reactant

@traceable read_entry(table, row, column) = table[row, column]
function subtotal(table, x, row, count)
    table, x, row, count = map(ReactiveKernels._loop_capture_traced,
        (table, x, row, count))
    result = zero(x)
    i = 0
    Reactant.@trace checkpointing=Reactant.Binomial(4) while i < count
        j = i + 1
        result += log1p(ReactiveKernels.traced(read_entry, table, row, j))
        i += 1
    end
    result
end
function valid_count(table, x, row, count)
    result = zero(x)
    Reactant.@trace if count > size(table, 2)
        result = zero(x)
    else
        result = subtotal(table, x, row, count)
    end
    result
end
function guarded_sum(table, x, row, count)
    table, x, row, count = map(ReactiveKernels._loop_capture_traced,
        (table, x, row, count))
    result = zero(x)
    Reactant.@trace if x < 0
        result = zero(x)
    else
        result = valid_count(table, x, row, count)
    end
    result
end
@kernel matrix_plate(x::AbstractVector, rows, counts, table::AbstractMatrix) = begin
    values = plate(x, rows, counts, Ref(table)) do xi, row, count, whole
        guarded_sum(whole, xi, row, count) * xi + xi
    end
    total = sum(values)
    return total
end
end

function _shared_array_mlir_inventory(text)
    ops = [m.match for m in eachmatch(
        r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.[a-zA-Z_]+", text)]
    Dict(op => count(==(op), ops) for op in unique(ops))
end
function _shared_array_xla_inventory(text)
    ops = [m.captures[1] for m in eachmatch(
        r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", text)]
    Dict(op => count(==(op), ops) for op in unique(ops))
end

@testset "shared host matrices retain nested lazy branches and loops" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt) !== nothing
    for T in (Float32, Float64)
        inventories = []
        for n in (8, 16, 64, 128)
            table = T[0.1 + 0.01 * (i + j) for i in 1:n, j in 1:2]
            x = T[i % 5 == 0 ? -0.25 : 0.25 for i in 1:n]
            rows = collect(1:n)
            counts = [(0, 2, 3, 2)[mod1(i, 4)] for i in 1:n]
            for i in 1:n
                if x[i] < 0 || counts[i] > size(table, 2)
                    table[i, :] .= -2
                end
            end
            saved = deepcopy((table, rows, counts))
            kernel = prepare(SharedHostArrayBranchFixtures.matrix_plate;
                want=:total, bound=(; table, rows, counts))
            gradient(v) = only(Enzyme.gradient(Enzyme.Reverse, kernel, v))
            rx = Reactant.to_rarray(x)
            primal = Reactant.@compile kernel(rx)
            reverse = Reactant.@compile gradient(rx)
            changed_guard = copy(x)
            changed_guard[2] = -changed_guard[2]
            for v in (x, 2x, changed_guard)
                rv = Reactant.to_rarray(v)
                expected = sum(v) + sum((v[i] * sum(log1p(table[i, j])
                    for j in 1:counts[i]; init=zero(T)) for i in 1:n
                    if v[i] >= 0 && counts[i] <= size(table, 2)); init=zero(T))
                @test kernel(v) ≈ expected rtol=1e-6
                @test Float64(primal(rv)) ≈ expected rtol=1e-6
                @test Array(reverse(rv)) ≈ gradient(v) rtol=1e-6
                expected_gradient = T[1 + (v[i] >= 0 && counts[i] <= size(table, 2) ?
                    sum(log1p(table[i, j]) for j in 1:counts[i]; init=zero(T)) : zero(T))
                    for i in 1:n]
                @test gradient(v) ≈ expected_gradient rtol=1e-6
                @test Array(rv) == v
                @test (table, rows, counts) == saved
            end
            mlir = repr(Reactant.@code_hlo kernel(rx))
            hlo = repr(only(Reactant.XLA.get_hlo_modules(primal.exec)))
            reverse_mlir = repr(Reactant.@code_hlo gradient(rx))
            reverse_hlo = repr(only(Reactant.XLA.get_hlo_modules(reverse.exec)))
            inventory = (_shared_array_mlir_inventory(mlir),
                _shared_array_xla_inventory(hlo),
                _shared_array_mlir_inventory(reverse_mlir),
                _shared_array_xla_inventory(reverse_hlo))
            # The outer plate and inner data-count loop, both lazy guards and
            # the sole nonlinear body survive the default executable.
            @test get(inventory[1], "stablehlo.while", 0) == 2
            @test get(inventory[1], "stablehlo.if", 0) == 2
            @test get(inventory[2], "while", 0) == 2
            @test get(inventory[2], "conditional", 0) == 2
            @test get(inventory[2], "log-plus-one", 0) == 1
            push!(inventories, inventory)
            if haskey(ENV, "RK_SHARED_ARRAY_EVIDENCE_DIR")
                dir = ENV["RK_SHARED_ARRAY_EVIDENCE_DIR"]
                for (name, text) in (("primal.mlir", mlir), ("primal.hlo", hlo),
                        ("reverse.mlir", reverse_mlir), ("reverse.hlo", reverse_hlo))
                    write(joinpath(dir, "$(T)-$(n)-$(name)"), text)
                end
            end
        end
        # Small plates and XLA's larger reduction strategy each have a fixed
        # complete operation inventory; doubling within either adds no ops.
        @test inventories[1] == inventories[2]
        @test inventories[3] == inventories[4]
    end
    for T in (Float32, Float64)
        kernel = prepare(SharedHostArrayBranchFixtures.matrix_plate; want=:total,
            bound=(table=zeros(T, 2, 2), rows=Int[], counts=Int[]))
        x = T[]
        rx = Reactant.to_rarray(x)
        primal = Reactant.@compile kernel(rx)
        @test kernel(x) == zero(T)
        @test Float64(primal(rx)) == 0
        @test isempty(Array(rx))
    end
end
