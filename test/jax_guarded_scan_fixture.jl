@kernel function jax_guarded_scan_rows(a::Float64, bases::Vector{Vector{Float64}},
                                      types::Vector{Vector{Int}}, orders::Vector{Vector{Int}})
    subjects = plate(bases, types, orders) do base, kinds, visits
        offsets = plate(base) do x
            (a * x, 2a * x)
        end
        path = scan(kinds, visits, Ref(offsets); init = 0.5) do carry, kind, j, table
            moved = 0.75 * carry + a
            next = kind == 3 ? moved + table[j][1] : moved
            (next, next)
        end
        sum(path)
    end
    total = sum(subjects)
    return total
end
