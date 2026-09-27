# data: y x z
begin
    eta = a .+ b .* x
    ls = c .+ d .* z
    y .~ InverseGaussian.(exp.(eta), exp.(ls))
end
