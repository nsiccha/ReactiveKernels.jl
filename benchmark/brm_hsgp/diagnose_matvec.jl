# Control for the model's two projections and their two reverse projections.
# This quadratic is not a replacement density; it isolates dense linear algebra.
include(joinpath(@__DIR__, "diagnose.jl"))

function projection_primal(A, q, c)
    u, v = A * q[3:22], A * q[25:44]
    sum(abs2, u) + sum(abs2, v) + sum(abs2, q[[1,2,23,24]]) + sum(c)
end

function projection_value_gradient(A, q, c)
    u, v = A * q[3:22], A * q[25:44]
    value = sum(abs2, u) + sum(abs2, v) + sum(abs2, q[[1,2,23,24]]) + sum(c)
    gradient = vcat(2 .* q[1:2], 2 .* (transpose(A) * u),
        2 .* q[23:24], 2 .* (transpose(A) * v))
    value, gradient
end

# The same quadratic, grouping the two right-hand sides into one matrix.
function grouped_projection_primal(A, q, c)
    U = A * hcat(q[3:22], q[25:44])
    sum(abs2, U) + sum(abs2, q[[1,2,23,24]]) + sum(c)
end

function grouped_projection_value_gradient(A, q, c)
    U = A * hcat(q[3:22], q[25:44])
    value = sum(abs2, U) + sum(abs2, q[[1,2,23,24]]) + sum(c)
    G = 2 .* (transpose(A) * U)
    value, vcat(2 .* q[1:2], G[:,1], 2 .* q[23:24], G[:,2])
end

function matrix_diagnostics(bundle, output)
    mkpath(output)
    BLAS.set_num_threads(1)
    data = BRMHSGPExample.motorcycle_data(joinpath(@__DIR__, "..", "..", "examples", "data", "mcycle.csv"))
    q = copy(deserialize(joinpath(bundle,"noncentered.jls")).posterior_position[:,5000])
    c = zeros(40)
    A = BRMHSGPExample.prepare_model(data; want=:basis)(q,c)
    primal = (q,c) -> projection_primal(A,q,c)
    vg = (q,c) -> projection_value_gradient(A,q,c)
    rq, rc = Reactant.to_rarray(q), Reactant.to_rarray(c)
    rp = Reactant.@compile sync=true primal(rq,rc)
    rvg = Reactant.@compile sync=true vg(rq,rc)
    grouped_primal = (q,c) -> grouped_projection_primal(A,q,c)
    grouped_vg = (q,c) -> grouped_projection_value_gradient(A,q,c)
    grouped_rp = Reactant.@compile sync=true grouped_primal(rq,rc)
    grouped_rvg = Reactant.@compile sync=true grouped_vg(rq,rc)
    v,g = vg(q,c)
    rv,rg = rvg(rq,rc)
    @assert isapprox(Float64(rv),v; rtol=2e-12,atol=2e-12)
    @assert isapprox(Array(rg),g; rtol=2e-12,atol=2e-12)
    gv,gg = grouped_rvg(rq,rc)
    @assert isapprox(Float64(gv),v; rtol=2e-12,atol=2e-12)
    @assert isapprox(Array(gg),g; rtol=2e-12,atol=2e-12)
    result = Dict("source_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "native_primal"=>measure(() -> primal(q,c)),
        "native_value_gradient"=>measure(() -> vg(q,c)),
        "reactant_primal"=>measure(() -> rp(rq,rc)),
        "reactant_value_gradient"=>measure(() -> rvg(rq,rc)),
        "reactant_grouped_primal"=>measure(() -> grouped_rp(rq,rc)),
        "reactant_grouped_value_gradient"=>measure(() -> grouped_rvg(rq,rc)))
    write(joinpath(output,"projection.mlir"), String(Reactant.@code_hlo optimize=true vg(rq,rc)))
    write(joinpath(output,"grouped-projection.mlir"), String(Reactant.@code_hlo optimize=true grouped_vg(rq,rc)))
    open(joinpath(output,"receipt.toml"),"w") do io
        TOML.print(io,result;sorted=true)
    end
    println("matrix controls complete")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: diagnose_matvec.jl BUNDLE_DIR OUTPUT_DIR")
    matrix_diagnostics(ARGS...)
end
