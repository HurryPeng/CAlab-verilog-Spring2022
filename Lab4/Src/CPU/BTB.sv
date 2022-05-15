// `define STRATEGY_ZERO
`define STRATEGY_BTB
// `define STRATEGY_BHT

module BTB #(
    parameter  SIZE_LEN = 4
)(
    input wire [31:0] predictPc,
    output reg predict,
    output reg [31:0] predictTarget,

    input wire clk,
    input wire rst,
    input wire update,
    input wire br,
    input wire [31:0] updatePc,
    input wire [31:0] updateTarget
);

    localparam SIZE = 1 << SIZE_LEN;
    localparam TAG_LEN = 32 - SIZE_LEN;

    reg [31:0] btbMem [SIZE];
    reg [TAG_LEN-1:0] btbTags [SIZE];
    reg valid [SIZE];
    reg [1:0] bhtMem [SIZE];

    wire [TAG_LEN-1:0] predictTagAddr;
    wire [SIZE_LEN-1:0] predictSetAddr;
    assign {predictTagAddr, predictSetAddr} = predictPc;

    wire [TAG_LEN-1:0] updateTagAddr;
    wire [SIZE_LEN-1:0] updateSetAddr;
    assign {updateTagAddr, updateSetAddr} = updatePc;

    always @* begin
        predict = 0;
        predictTarget = 0;

        if (valid[predictSetAddr] && btbTags[predictSetAddr] == predictTagAddr) begin
            `ifdef STRATEGY_ZERO
                predict = 0;
            `elsif STRATEGY_BTB
                predict = 1;
            `elsif STRATEGY_BHT
                predict = (bhtMem[predictSetAddr] > 1);
            `endif

            predictTarget = btbMem[predictSetAddr];
        end
    end

    always @(posedge clk, posedge rst) begin
        if (rst) begin
            for (integer i = 0; i < SIZE; i++) begin
                btbMem[i] <= 0;
                btbTags[i] <= 0;
                valid[i] <= 0;
                bhtMem[i] <= 0;
            end
        end
        else begin
            if (update) begin
                `ifdef STRATEGY_BTB
                    if (br) begin
                        btbMem[updateSetAddr] <= updateTarget;
                        btbTags[updateSetAddr] <= updateTagAddr;
                        valid[updateSetAddr] <= 1;
                    end
                    else begin
                        btbMem[updateSetAddr] <= 0;
                        btbTags[updateSetAddr] <= 0;
                        valid[updateSetAddr] <= 0;
                    end
                `elsif STRATEGY_BHT
                    if (br) begin
                        btbMem[updateSetAddr] <= updateTarget;
                        btbTags[updateSetAddr] <= updateTagAddr;
                        valid[updateSetAddr] <= 1;
                        if (valid[updateSetAddr] && btbTags[updateSetAddr] == updateTagAddr) begin
                            if (bhtMem[updateSetAddr] < 3) bhtMem[updateSetAddr] <= bhtMem[updateSetAddr] + 1;
                        end
                        else begin
                            bhtMem[updateSetAddr] <= 2;
                        end
                    end
                    else begin
                        if (valid[updateSetAddr] && btbTags[updateSetAddr] == updateTagAddr) begin
                            if (bhtMem[updateSetAddr] > 0) bhtMem[updateSetAddr] <= bhtMem[updateSetAddr] - 1;
                        end
                    end
                `endif
            end
        end
    end

endmodule
