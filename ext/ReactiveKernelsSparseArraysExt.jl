# A bound sparse left factor's rows, for a cell's product over it
# (`src/cell_sparse_products.jl`).
module ReactiveKernelsSparseArraysExt

using ReactiveKernels: ReactiveKernels, _CellSparseRows
using SparseArrays: AbstractSparseArray, SparseMatrixCSC, getcolptr, nonzeros, rowvals

ReactiveKernels._cell_sparse_type(::Type{<:AbstractSparseArray}) = true

# The compressed columns of the transpose are the matrix's rows, each ascending,
# with every stored entry kept, explicit zeros included.
function ReactiveKernels._cell_sparse_rows(A::SparseMatrixCSC)
    rows = copy(transpose(A))
    _CellSparseRows(A, size(A, 1), size(A, 2), Vector{Int}(getcolptr(rows)),
                    Vector{Int}(rowvals(rows)), nonzeros(rows))
end

end
