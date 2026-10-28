# data: y g o
begin
    c[levels(g)] .~ Normal.(0, 2)
    mu = c[g] .+ o
    y .~ Normal.(mu, 1.5)
end
