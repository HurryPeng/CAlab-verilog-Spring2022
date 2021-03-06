`timescale 1ns / 1ps
//  功能说明
    //  将Cache中Load的数据扩展成32位
// 输入
    // data              cache读出的数据
    // addr              字节地址
    // load_type         load的类型
// 输出
    // dealt_data        扩展完的数据
// 实验要求
    // 补全模块


`include "Parameters.v"

module DataExtend(
    input wire [31:0] data,
    input wire [1:0] addr,
    input wire [2:0] load_type,
    output reg [31:0] dealt_data
    );

    // TODO: Complete this module

    wire [31:0] shifted_data;
    assign shifted_data = data >> (8 * addr);
    
    always @ (*)
    begin
        /* FIXME: Write your code here... */
        
        case (load_type)
            `NOREGWRITE:dealt_data = data;
            `LW:        dealt_data = shifted_data;
            `LH:        dealt_data = {{16{shifted_data[15]}}, shifted_data[15:0]};
            `LHU:       dealt_data = {16'b0, shifted_data[15:0]};
            `LB:        dealt_data = {{24{shifted_data[7]}}, shifted_data[7:0]};
            `LBU:       dealt_data = {24'b0, shifted_data[7:0]};
            default:    dealt_data = 0;
        endcase
    end

endmodule