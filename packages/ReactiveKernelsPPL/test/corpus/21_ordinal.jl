# data: y x
begin
    b ~ Normal(0, 1)
    eta = b .* x
    y_thresholds[1:length(levels(y)) - 1] .~ Normal.(0, 1)
    y .~ Ordinal.(StoppingRatio(), ProbitLink(), eta, Ref(y_thresholds))
end
