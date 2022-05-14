module BTB #(
    parameter  SIZE_LEN = 4
)(
    input wire [31:0] predictPc,
    output reg predict,
    output reg [31:0] predictTarget,

    input wire clk,
    input wire rst,
    input wire update,
    input wire set,
    input wire [31:0] updatePc,
    input wire [31:0] updateTarget
);

    localparam SIZE = 1 << SIZE_LEN;
    localparam TAG_LEN = 32 - SIZE_LEN;

    reg [31:0] btbMem [SIZE];
    reg [TAG_LEN-1:0] btbTags [SIZE];
    reg valid [SIZE];

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
            predict = 1;
            predictTarget = btbMem[predictSetAddr];
        end
    end

    always @(posedge clk, posedge rst) begin
        if (rst) begin
            for (integer i = 0; i < SIZE; i++) begin
                btbMem[i] <= 0;
                btbTags[i] <= 0;
                valid[i] <= 0;
            end
        end
        else begin
            if (update) begin
                if (set) begin
                    btbMem[updateSetAddr] <= updateTarget;
                    btbTags[updateSetAddr] <= updateTagAddr;
                    valid[updateSetAddr] <= 1;
                end
                else begin
                    btbMem[updateSetAddr] <= 0;
                    btbTags[updateSetAddr] <= 0;
                    valid[updateSetAddr] <= 0;
                end
            end
        end
    end

endmodule
