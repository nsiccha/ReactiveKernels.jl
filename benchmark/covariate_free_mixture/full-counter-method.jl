function native_counts(f)
    counts = zeros(UInt64, 7)
    f(); f()
    ccall(:begin_count, Cvoid, ())
    f()
    ccall(:end_counts, Cvoid, (Ptr{UInt64},), counts)
    (; malloc=counts[1], calloc=counts[2], realloc=counts[3], free=counts[4],
        requested_bytes=(malloc=counts[5], calloc=counts[6], realloc=counts[7]))
end
