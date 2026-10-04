module SubmodelKeywordTests
using ReactiveKernelsPPL, Test

@rkppl shifted(x; predictor = 2.0) = begin
    return x .+ predictor
end

@rkppl stream(x) = begin
    slot .~ Normal.(x, 1)
    return slot
end

# A retired catalogue spelling remains an ordinary caller-defined submodel.
@rkppl LKJCovarianceFactor(x) = begin
    return x .+ 1
end

@testset "submodel keywords have their declared meaning" begin
    data = (; x = [0.1, 0.4], y = [0.2, -0.3])
    ordinary = lower_rkppl(quote
        mu ~ shifted(x; predictor = 0.5)
        y .~ Normal.(mu, 1)
    end, data; mod = @__MODULE__, conditioned = (:y,))
    @test validate_structure(ordinary) === nothing
    renamed = lower_rkppl(quote
        mu ~ LKJCovarianceFactor(x)
        y .~ Normal.(mu, 1)
    end, data; mod = @__MODULE__, conditioned = (:y,))
    @test validate_structure(renamed) === nothing

    # Refused: USER 1cmodra (names) removes the undeclared predictor pin.
    # A declared keyword named predictor remains an ordinary argument above.
    err = try
        lower_rkppl(quote
            y ~ stream(x; predictor = mu)
        end, data; mod = @__MODULE__, conditioned = (:y,))
    catch error
        error
    end
    @test err isa SurfaceLoweringError
    @test occursin("unknown keyword `predictor`", sprint(showerror, err))
    @test occursin("ordinary assignments or declarations", sprint(showerror, err))
end
end
