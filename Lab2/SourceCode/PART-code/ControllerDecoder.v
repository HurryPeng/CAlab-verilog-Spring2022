`timescale 1ns / 1ps
//  功能说明
    //  对指令进行译码，将其翻译成控制信号，传输给各个部件
// 输入
    // Inst              待译码指令
// 输出
    // jal               jal跳转指令
    // jalr              jalr跳转指令
    // op1_src           0表示ALU操作数1来自寄存器，1表示来自PC-4
    // op2_src           ALU的第二个操作数来源。为1时，op2选择imm，为0时，op2选择reg2
    // ALU_func          ALU执行的运算类型
    // br_type           branch的判断条件，可以是不进行branch
    // load_npc          写回寄存器的值的来源（PC或者ALU计算结果）, load_npc == 1时选择PC
    // wb_select         写回寄存器的值的来源（Cache内容或者ALU计算结果），wb_select == 1时选择cache内容
    // load_type         load类型
    // reg_write_en      通用寄存器写使能，reg_write_en == 1表示需要写回reg
    // cache_write_en    按字节写入data cache
    // imm_type          指令中立即数类型
    // CSR_write_en
    // CSR_zimm_or_reg
// 实验要求
    // 补全模块


`include "Parameters.v"   
module ControllerDecoder(
    input wire [31:0] inst,
    output reg jal,
    output reg jalr,
    output reg op1_src, op2_src,
    output reg [3:0] ALU_func,
    output reg [2:0] br_type,
    output reg load_npc,
    output reg wb_select,
    output reg [2:0] load_type,
    output reg reg_write_en,
    output reg [3:0] cache_write_en,
    output reg [2:0] imm_type,
    // CSR signals
    output reg CSR_write_en,
    output reg CSR_zimm_or_reg
    );

    // TODO: Complete this module

    wire [6:0] opcode, funct7;
    wire [2:0] funct3;

    assign opcode = inst[6:0];
    assign funct7 = inst[31:25];
    assign funct3 = inst[14:12];

    always @ (*) begin
        ALU_func = 0; // [3:0]
        imm_type = 0; // [2:0]

        op1_src = 0;
        op2_src = 0;

        reg_write_en = 0;
        load_npc = 0;
        wb_select = 0;

        br_type = 0; // [2:0]
        jal = 0;
        jalr = 0;

        load_type = 0; // [2:0]
        cache_write_en = 0; // [3:0]

        CSR_write_en = 0;
        CSR_zimm_or_reg = 0;

        /* FIXME: Write your code here... */

        case (opcode)
            `U_LUI: begin
                ALU_func = `LUI;
                imm_type = `UTYPE;

                op2_src = 1;

                reg_write_en = 1;
            end

            `U_AUIPC: begin
                ALU_func = `ADD;
                imm_type = `UTYPE;

                op1_src = 1;
                op2_src = 1;

                reg_write_en = 1;
            end
            
            `J_JAL: begin
                imm_type = `JTYPE;

                reg_write_en = 1;
                load_npc = 1;
                
                jal = 1;
            end

            `B_TYPE: begin

                imm_type = `BTYPE;

                op1_src = 0;
                op2_src = 0;

                br_type = `NOBRANCH;

                case (funct3)
                    `B_BEQ: br_type = `BEQ;
                    `B_BNE: br_type = `BNE;
                    `B_BLT: br_type = `BLT;
                    `B_BGE: br_type = `BGE;
                    `B_BLTU:br_type = `BLTU;
                    `B_BGEU:br_type = `BGEU;
                endcase
                
            end

            `I_LOAD: begin
                ALU_func = `ADD;
                imm_type = `ITYPE;

                op2_src = 1;

                reg_write_en = 1;
                wb_select = 1;
                
                case (funct3)
                    `I_LW:  load_type = `LW;
                    `I_LB:  load_type = `LB;
                    `I_LH:  load_type = `LH;
                    `I_LBU: load_type = `LBU;
                    `I_LHU: load_type = `LHU;
                    default: begin
                        reg_write_en = 0;
                        load_type = `NOREGWRITE;
                    end
                endcase
            end

            `I_ARI: begin
                ALU_func = 0;
                imm_type = `ITYPE;

                op1_src = 0;
                op2_src = 1;

                reg_write_en = 1;

                case (funct3)
                    `I_ADDI:    ALU_func = `ADD;
                    `I_SLTI:    ALU_func = `SLT;
                    `I_SLTIU:   ALU_func = `SLTU;
                    `I_XORI:    ALU_func = `XOR;
                    `I_ORI:     ALU_func = `OR;
                    `I_ANDI:    ALU_func = `AND;
                    `I_SLLI:    ALU_func = `SLL;
                    `I_SR: begin
                        case (funct7)
                            `I_SRAI:    ALU_func = `SRA;
                            `I_SRLI:    ALU_func = `SRL;
                        endcase
                    end
                endcase
            end

            `I_JALR: begin
                ALU_func = `ADD;
                imm_type = `ITYPE;

                op1_src = 0;
                op2_src = 1;

                reg_write_en = 1;
                load_npc = 1;

                jalr = 1;
            end

            `S_TYPE: begin
                ALU_func = `ADD;
                imm_type = `STYPE;

                op1_src = 0;
                op2_src = 1;

                cache_write_en = 0;

                case (funct3)
                    `S_SB: cache_write_en = 4'b0001;
                    `S_SH: cache_write_en = 4'b0011;
                    `S_SW: cache_write_en = 4'b1111;
                endcase
            end

            `R_TYPE: begin
                ALU_func = 0;

                op1_src = 0;
                op2_src = 0;

                reg_write_en = 1;

                case (funct3)
                    `R_AS: begin
                        case (funct7)
                            `R_ADD: ALU_func = `ADD;
                            `R_SUB: ALU_func = `SUB;
                        endcase
                    end
                    `R_SLL: ALU_func = `SLL;
                    `R_SLT: ALU_func = `SLT;
                    `R_SLTU:ALU_func = `SLTU;
                    `R_XOR: ALU_func = `XOR;
                    `R_SR: begin
                        case (funct7)
                            `R_SRL: ALU_func = `SRL;
                            `R_SRA: ALU_func = `SRA;
                        endcase
                    end
                    `R_OR:  ALU_func = `OR;
                    `R_AND: ALU_func = `AND;
                endcase
            end

            `I_CSR: begin
                ALU_func = 0;

                op1_src = 0;
                op2_src = 0;

                reg_write_en = 1;

                CSR_write_en = 1;
                CSR_zimm_or_reg = 0;
                
                case (funct3)
                    `I_CSRRC:  begin
                        ALU_func = `NOR;
                        CSR_zimm_or_reg = 0;
                    end  
                    `I_CSRRCI: begin
                        ALU_func = `NOR;
                        CSR_zimm_or_reg = 1;
                    end  
                    `I_CSRRS:  begin
                        ALU_func = `OR;
                        CSR_zimm_or_reg = 0;
                    end  
                    `I_CSRRSI: begin
                        ALU_func = `OR;
                        CSR_zimm_or_reg = 1;
                    end  
                    `I_CSRRW:  begin
                        ALU_func = `OP1;
                        CSR_zimm_or_reg = 0;
                    end  
                    `I_CSRRWI: begin
                        ALU_func = `OP1;
                        CSR_zimm_or_reg = 1;
                    end  
                endcase
            end
        endcase
    end

endmodule