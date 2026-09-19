# data: y x
begin
    eta = a .+ b .* x
    y .~ OrderedLogistic.(eta)
end
