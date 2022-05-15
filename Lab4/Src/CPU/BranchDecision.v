`timescale 1ns / 1ps
//  功能说明
    //  判断是否branch
// 输入
    // reg1               寄存器1
    // reg2               寄存器2
    // br_type            branch类型
// 输出
    // br                 是否branch
// 实验要求
    // 补全模块

`include "Parameters.v"   
module BranchDecision(
    input wire predict,
    input wire [31:0] pc,
    input wire [31:0] br_target,
    input wire [31:0] reg1, reg2,
    input wire [2:0] br_type,
    output reg br,
    output reg brFlush,
    output reg [31:0] brFlushTarget
    );

    // TODO: Complete this module
    wire [32:0] diffExt;
    assign diffExt = { reg1[31], reg1 } - { reg2[31], reg2 };


    always @ (*)
    begin
        case(br_type)
            /* FIXME: Write your code here... */

            `NOBRANCH:  br = 0;
            `BEQ:       br = (reg1 == reg2) ? 1 : 0;
            `BNE:       br = (reg1 != reg2) ? 1 : 0;
            `BLT:       br = diffExt[32];
            `BLTU:      br = (reg1 < reg2) ? 1 : 0;
            `BGE:       br = !diffExt[32];
            `BGEU:      br = (reg1 >= reg2) ? 1 : 0;
            default:    br = 0;
        endcase
    end

    always @* begin
        brFlush = 0;
        brFlushTarget = 0;
        if (br != predict) begin
            brFlush = 1;
            if (br) begin
                brFlushTarget = br_target;
            end
            else begin
                brFlushTarget = pc + 4;
            end
        end
    end

endmodule
