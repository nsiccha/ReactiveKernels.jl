using ReactiveKernels, Reactant, Test
import Enzyme
Reactant.set_default_backend("cpu")

# Dispatch on the view is observable: converting a capture to an owned array
# would make these calls fail even when its elements were equal.
_loop_view_read(w::SubArray) = sum(w)

@traceable function loop_view_sum(x, n)
    w = view(x, 2:2:length(x))
    s = copy(x)
    for i in 1:n
        s = s .+ _loop_view_read(w)
    end
    sum(s) + sum(x)
end

@traceable function loop_nested_view_sum(x)
    w = view(view(x, 2:length(x)), 1:2:(length(x) - 1))
    captured = (; w, host = [0.25])
    acc = 0.0
    for i in eachindex(x)
        acc = if x[i] > 0
            acc + _loop_view_read(captured.w) + captured.host[1]
        else
            acc
        end
    end
    acc + sum(x)
end

@traceable function loop_matrix_view_sum(x)
    w = view(x, 2:2:size(x, 1), :)
    col = view(x, :, 2)
    acc = 0.0
    i = 1
    while i <= size(x, 1)
        acc = acc + _loop_view_read(w) + _loop_view_read(col)
        i = i + 1
    end
    acc + sum(x)
end

@kernel scan_shared_view(x) = begin
    w = view(x, 2:2:length(x))
    prefix = scan(x, Ref(w); init = 0.0) do acc, xi, shared
        next = acc + xi + _loop_view_read(shared)
        (next, next)
    end
    total = sum(prefix) + sum(x)
    return total
end

function _loop_view_inventory(hlo)
    ops = [m.match for m in eachmatch(r"\b(?:stablehlo|enzyme|arith|tensor)\.[a-zA-Z_]+", hlo)]
    Dict(op => count(==(op), ops) for op in unique(ops))
end

@testset "view captures preserve primal, reverse and fixed loop structure" begin
    functions = (
        x -> ReactiveKernels.traced(loop_view_sum, x, length(x)),
        x -> ReactiveKernels.traced(loop_nested_view_sum, x),
        x -> ReactiveKernels.traced(loop_matrix_view_sum, x),
        prepare(scan_shared_view),
    )
    native_functions = (
        x -> loop_view_sum(x, length(x)),
        loop_nested_view_sum,
        loop_matrix_view_sum,
        prepare(scan_shared_view),
    )
    for (index, (f, native)) in enumerate(zip(functions, native_functions))
        inventories = []
        reverse_inventories = []
        for n in (8, 32)
            x = index == 3 ? reshape(sin.(1:(3n)), n, 3) : sin.(1:n)
            saved = copy(x)
            rx = Reactant.to_rarray(x)
            exe = Reactant.compile(f, (rx,))
            @test Float64(exe(rx)) ≈ native(x) rtol=2e-13 atol=2e-13
            @test isequal(x, saved)
            @test isequal(Array(rx), saved)

            gradient(v) = Enzyme.gradient(Enzyme.Reverse, f, v)
            native_gradient = only(Enzyme.gradient(Enzyme.Reverse, native, x))
            reverse = Reactant.compile(gradient, (rx,))
            @test Array(only(reverse(rx))) ≈ native_gradient rtol=2e-13 atol=2e-13
            @test isequal(x, saved)
            @test isequal(Array(rx), saved)

            hlo = repr(Reactant.@code_hlo f(rx))
            # The default optimizer may replace an affine recurrence by its
            # closed form. It must not duplicate regions as data lengths grow.
            @test count("stablehlo.while", hlo) <= 1
            raw_hlo = repr(Reactant.@code_hlo optimize = false f(rx))
            @test count("stablehlo.while", raw_hlo) == 1
            push!(inventories, _loop_view_inventory(hlo))
            reverse_hlo = repr(Reactant.@code_hlo gradient(rx))
            push!(reverse_inventories, _loop_view_inventory(reverse_hlo))
        end
        @test inventories[1] == inventories[2]
        @test reverse_inventories[1] == reverse_inventories[2]
    end
end

@testset "empty view and zero trip count" begin
    for x in (Float64[], [0.2, 0.4])
        rx = Reactant.to_rarray(x)
        f(v) = ReactiveKernels.traced(loop_view_sum, v, 0)
        exe = Reactant.compile(f, (rx,))
        @test Float64(exe(rx)) == 2sum(x)
        @test isequal(Array(rx), x)
    end
end

@testset "read-only views cross retained recipe loops" begin
    x = [0.2, -0.1, 0.3, 0.9, -0.2, 0.4, 0.1, 0.7]
    saved = copy(x)
    rx = Reactant.to_rarray(x)
    f(v) = ReactiveKernels.traced(loop_view_sum, v, length(v))
    exe = Reactant.compile(f, (rx,))
    expected = 2sum(x) + length(x)^2 * sum(x[2:2:end])
    @test Float64(exe(rx)) ≈ expected
    @test isequal(x, saved)
    @test isequal(Array(rx), saved)
end
