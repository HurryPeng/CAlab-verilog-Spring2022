`timescale 1ns / 1ps

`include "Parameters.v"   
module BranchStats(
    input wire clk,
    input wire rst,
    input wire bubbleE,
    input wire [2:0] br_type,
    input wire predict,
    input wire brFlush
    );

    reg [31:0] stats_branch;
    reg [31:0] stats_branch_miss;

    always @(posedge clk, posedge rst) begin
        if (rst) begin
            stats_branch <= 0;
            stats_branch_miss <= 0;
        end
        else begin
            if (!bubbleE) begin
                if (br_type != `NOBRANCH) begin
                    stats_branch <= stats_branch + 1;
                end
                if (brFlush) begin
                    stats_branch_miss <= stats_branch_miss + 1;
                end
            end
        end
    end

endmodule
