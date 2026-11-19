using Test, ReactiveKernels, Enzyme

isdefined(@__MODULE__, :MatrixRecordPlateFixture) || include("fixtures/matrix_record_plate.jl")

@kernel named_matrix_records(x::Float64, a::Vector{Int}, b::Vector{Int}, y::Matrix{Float64}) = begin
    states = collect(Iterators.product(a, b))
    records = plate(states, x, Ref(y)) do ij, s, ys
        (; pointwise = (; y = ys .* s), prior = s * ij[1] + ij[2])
    end
    terms = plate(records) do r
        sum(r.pointwise.y) + r.prior
    end
    value = sum(terms) + length(records) + size(records, 2)
    return value
end

@testset "ordinary native Reverse of matrix-valued record fields" begin
    F = MatrixRecordPlateFixture
    prepare_value(spec) = prepare(spec; have = (:x, :a, :b, :y), want = :value)
    readers = (mixed = prepare_value(F.matrix_records), named = prepare_value(named_matrix_records))
    # Registered 0.13.221 on Julia 1.10 rejects even the comprehension. Older
    # dogfood at the same version accepts it but rejects a nonempty record
    # store. Exercise both paths and report only their observed exceptions.
    legacy = VERSION.major == 1 && VERSION.minor == 10 && pkgversion(Enzyme) == v"0.13.221" &&
        !(isdefined(Enzyme, :has_dogfood_repair) && Enzyme.has_dogfood_repair(:mixed_record_merges))
    function check_gradient(f, x, a, b, y, expected, legacy_error)
        result = try
            Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
                Enzyme.Active(x), Enzyme.Const(a), Enzyme.Const(b), Enzyme.Const(y))[1][1]
        catch err
            legacy && err isa legacy_error || rethrow()
            err
        end
        @test result isa Real && result ≈ expected broken=(result isa Exception)
    end

    cases = (
        ([1, 2], [3, 4, 5], [0.2 0.3; 0.5 0.7]),
        (Int[], [1, 2], ones(2, 3)),
        ([1, 2], Int[], ones(2, 3)),
        (Int[], Int[], zeros(0, 0)),
        ([2, -3], [3], zeros(0, 3)),
        ([1, 2], [-3, 4], zeros(2, 0)),
        ([0, -2], [9, 7], zeros(0, 0)),
        (collect(1:8), collect(-3:6), [0.2 0.5; 0.3 0.7; -0.4 0.1]),
    )
    for (a, b, y) in cases
        @testset "mapped axes $((length(a), length(b))), field $(size(y))" begin
            x = 0.4
            value = F.reference(x, a, b, y)
            expected = length(b) * (length(a) * sum(y) + sum(a))
            check_gradient(F.reference, x, a, b, y, expected, Enzyme.Compiler.EnzymeRuntimeActivityError)
            store_error = isempty(a) || isempty(b) ? Union{} : Enzyme.Compiler.EnzymeNoTypeError
            for (layout, native) in pairs(readers)
                @testset "$layout" begin
                    @test native(x, a, b, y) ≈ value
                    check_gradient(native, x, a, b, y, expected, store_error)
                end
            end
        end
    end
end
