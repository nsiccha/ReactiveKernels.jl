# Public-safe independent IR control. Constant selection may remove a dead arm.
using Reactant, Test
const IR=Reactant.MLIR.IR
function source_module(n;altered=false)
    starts=join([2+3i for i in 0:n-1],", ")
    tags=altered ? "9, 9, 7, 6, 1, 2, 4, 1, 9" : "9, 1, 7, 6, 1, 2, 4, 1, 9"
    """
    module {
      func.func @table_selection_$n(%p: tensor<f64>) -> tensor<f64> {
        %starts = stablehlo.constant dense<[$starts]> : tensor<$(n)xi64>
        %tags = stablehlo.constant dense<[$tags]> : tensor<9xi64>
        %zero = stablehlo.constant dense<0> : tensor<i64>
        %one = stablehlo.constant dense<1> : tensor<i64>
        %three = stablehlo.constant dense<3> : tensor<i64>
        %limit = stablehlo.constant dense<$n> : tensor<i64>
        %fzero = stablehlo.constant dense<0.0> : tensor<f64>
        %outer:2 = stablehlo.while(%lane = %zero, %total = %fzero) : tensor<i64>, tensor<f64> attributes {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing, enzyme.checkpoint_period = 4 : i64} cond {
          %run = stablehlo.compare LT, %lane, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
          stablehlo.return %run : tensor<i1>
        } do {
          %position_array = stablehlo.dynamic_slice %starts, %lane, sizes = [1] : (tensor<$(n)xi64>, tensor<i64>) -> tensor<1xi64>
          %position = stablehlo.reshape %position_array : (tensor<1xi64>) -> tensor<i64>
          %index = stablehlo.subtract %position, %one : tensor<i64>
          %tag_array = stablehlo.dynamic_slice %tags, %index, sizes = [1] : (tensor<9xi64>, tensor<i64>) -> tensor<1xi64>
          %tag = stablehlo.reshape %tag_array : (tensor<1xi64>) -> tensor<i64>
          %selected = stablehlo.compare EQ, %tag, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
          %term = "stablehlo.if"(%selected) ({
            stablehlo.return %p : tensor<f64>
          }, {
            %inner:2 = stablehlo.while(%i = %zero, %sum = %fzero) : tensor<i64>, tensor<f64> attributes {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing, enzyme.checkpoint_period = 4 : i64} cond {
              %run = stablehlo.compare LT, %i, %three : (tensor<i64>, tensor<i64>) -> tensor<i1>
              stablehlo.return %run : tensor<i1>
            } do {
              %next_i = stablehlo.add %i, %one : tensor<i64>
              %next_sum = stablehlo.add %sum, %p : tensor<f64>
              stablehlo.return %next_i, %next_sum : tensor<i64>, tensor<f64>
            }
            stablehlo.return %inner#1 : tensor<f64>
          }) : (tensor<i1>) -> tensor<f64>
          %next_lane = stablehlo.add %lane, %one : tensor<i64>
          %next_total = stablehlo.add %total, %term : tensor<f64>
          stablehlo.return %next_lane, %next_total : tensor<i64>, tensor<f64>
        }
        return %outer#1 : tensor<f64>
      }
    }
    """
end
function counts!(d,op)
    name=IR.name(op);d[name]=get(d,name,0)+1
    for region in op,block in region,child in block
        counts!(d,child)
    end
    d
end
@testset "Constant table branch pruning retains requested runtime loops" begin
    for n in 1:3,altered in (false,true)
        IR.@with_context Reactant.ReactantContext() begin
            mod=parse(IR.Module,source_module(n;altered))
            try
                @test IR.verifyall(IR.Operation(mod);debug=true)
                before=counts!(Dict{String,Int}(),IR.Operation(mod))
                @test before["stablehlo.while"]==2
                passes=Reactant.Compiler.optimization_passes(Reactant.CompileOptions();backend="cpu")
                Reactant.Compiler.run_pass_pipeline!(mod,passes,"generic_constant_table")
                @test IR.verifyall(IR.Operation(mod);debug=true)
                after=counts!(Dict{String,Int}(),IR.Operation(mod))
                println("GENERIC CONSTANT TABLE n=",n," altered=",altered," COMPLETE DEFAULT INVENTORY ",sort!(collect(after)));flush(stdout)
                output_dir=get(ENV,"REACTANT_REPRO_OUTPUT","")
                if !isempty(output_dir)
                    mkpath(output_dir)
                    write(joinpath(output_dir,"generic-table-$(n)-$(altered).mlir"),repr(mod))
                end
                @test get(after,"stablehlo.while",0)==(n==1 && !altered ? 1 : 2)
                @test get(after,"stablehlo.if",0)==(n==1 ? 0 : 1)
            finally
                IR.dispose(mod)
            end
        end
    end
end
