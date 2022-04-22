`timescale 1ns / 1ps
// 功能说明
    // 同步读写Cache
    // debug端口用于simulation时批量读写数据，可以忽略
// 输入
    // clk               输入时钟
    // write_en          写使能
    // debug_write_en    debug写使能
    // addr              读写地址
    // debug_addr        debug读写地址
    // in_data           写入数据
    // debug_in_data     debug写入数据
    // out_data          输出数据
    // debug_out_data    debug输出数据
// 输出
    // douta             a口读数据
    // doutb             b口读数据
// 实验要求  
    // 无需修改


module DataCache(
    input wire clk,
    input wire [3:0] write_en, debug_write_en,
    input wire [31:2] addr, debug_addr,
    input wire [31:0] in_data, debug_in_data,
    output reg [31:0] out_data, debug_out_data
);


    // local variable
    wire addr_valid = (addr[31:14] == 18'h0);
    wire debug_addr_valid = (debug_addr[31:14] == 18'h0);
    wire [11:0] dealt_addr = addr[13:2];
    wire [11:0] dealt_debug_addr = debug_addr[13:2];
    // cache content
    reg [31:0] data_cache[0:4095];

    initial begin
        out_data = 32'h0;
        debug_out_data = 32'h0;
        // you can add simulation instructions here
        data_cache[       0] = 32'h00000076;
        data_cache[       1] = 32'h0000009c;
        data_cache[       2] = 32'h000000dc;
        data_cache[       3] = 32'h00000035;
        data_cache[       4] = 32'h00000046;
        data_cache[       5] = 32'h000000b4;
        data_cache[       6] = 32'h0000007e;
        data_cache[       7] = 32'h0000003d;
        data_cache[       8] = 32'h00000013;
        data_cache[       9] = 32'h0000001a;
        data_cache[      10] = 32'h00000053;
        data_cache[      11] = 32'h00000020;
        data_cache[      12] = 32'h000000ab;
        data_cache[      13] = 32'h0000001c;
        data_cache[      14] = 32'h000000bf;
        data_cache[      15] = 32'h00000040;
        data_cache[      16] = 32'h000000e8;
        data_cache[      17] = 32'h00000074;
        data_cache[      18] = 32'h0000003a;
        data_cache[      19] = 32'h000000bc;
        data_cache[      20] = 32'h00000097;
        data_cache[      21] = 32'h0000005f;
        data_cache[      22] = 32'h0000004f;
        data_cache[      23] = 32'h0000002c;
        data_cache[      24] = 32'h00000004;
        data_cache[      25] = 32'h00000009;
        data_cache[      26] = 32'h0000004a;
        data_cache[      27] = 32'h0000009e;
        data_cache[      28] = 32'h00000089;
        data_cache[      29] = 32'h000000a1;
        data_cache[      30] = 32'h00000014;
        data_cache[      31] = 32'h00000061;
        data_cache[      32] = 32'h000000c2;
        data_cache[      33] = 32'h000000ef;
        data_cache[      34] = 32'h0000000b;
        data_cache[      35] = 32'h00000029;
        data_cache[      36] = 32'h000000d1;
        data_cache[      37] = 32'h00000016;
        data_cache[      38] = 32'h000000d8;
        data_cache[      39] = 32'h00000028;
        data_cache[      40] = 32'h00000069;
        data_cache[      41] = 32'h000000a4;
        data_cache[      42] = 32'h000000fe;
        data_cache[      43] = 32'h0000000e;
        data_cache[      44] = 32'h000000c7;
        data_cache[      45] = 32'h000000b5;
        data_cache[      46] = 32'h00000036;
        data_cache[      47] = 32'h00000092;
        data_cache[      48] = 32'h000000f3;
        data_cache[      49] = 32'h000000f7;
        data_cache[      50] = 32'h000000e7;
        data_cache[      51] = 32'h00000045;
        data_cache[      52] = 32'h000000da;
        data_cache[      53] = 32'h000000a8;
        data_cache[      54] = 32'h000000ca;
        data_cache[      55] = 32'h0000003b;
        data_cache[      56] = 32'h0000006b;
        data_cache[      57] = 32'h00000025;
        data_cache[      58] = 32'h000000a7;
        data_cache[      59] = 32'h000000c0;
        data_cache[      60] = 32'h0000007c;
        data_cache[      61] = 32'h0000007f;
        data_cache[      62] = 32'h00000067;
        data_cache[      63] = 32'h000000b9;
        data_cache[      64] = 32'h000000ba;
        data_cache[      65] = 32'h000000ee;
        data_cache[      66] = 32'h000000f9;
        data_cache[      67] = 32'h0000009f;
        data_cache[      68] = 32'h000000b8;
        data_cache[      69] = 32'h0000005b;
        data_cache[      70] = 32'h00000024;
        data_cache[      71] = 32'h00000095;
        data_cache[      72] = 32'h000000a3;
        data_cache[      73] = 32'h00000034;
        data_cache[      74] = 32'h000000aa;
        data_cache[      75] = 32'h00000091;
        data_cache[      76] = 32'h00000038;
        data_cache[      77] = 32'h0000006c;
        data_cache[      78] = 32'h000000d7;
        data_cache[      79] = 32'h0000008e;
        data_cache[      80] = 32'h0000000a;
        data_cache[      81] = 32'h000000ea;
        data_cache[      82] = 32'h00000021;
        data_cache[      83] = 32'h0000005d;
        data_cache[      84] = 32'h00000041;
        data_cache[      85] = 32'h000000ec;
        data_cache[      86] = 32'h000000c5;
        data_cache[      87] = 32'h00000086;
        data_cache[      88] = 32'h000000f5;
        data_cache[      89] = 32'h00000019;
        data_cache[      90] = 32'h00000066;
        data_cache[      91] = 32'h0000002d;
        data_cache[      92] = 32'h00000062;
        data_cache[      93] = 32'h00000075;
        data_cache[      94] = 32'h000000d9;
        data_cache[      95] = 32'h000000ac;
        data_cache[      96] = 32'h00000060;
        data_cache[      97] = 32'h0000003c;
        data_cache[      98] = 32'h00000051;
        data_cache[      99] = 32'h00000068;
        data_cache[     100] = 32'h00000085;
        data_cache[     101] = 32'h0000008d;
        data_cache[     102] = 32'h000000d4;
        data_cache[     103] = 32'h000000d2;
        data_cache[     104] = 32'h000000cc;
        data_cache[     105] = 32'h000000be;
        data_cache[     106] = 32'h000000d5;
        data_cache[     107] = 32'h00000015;
        data_cache[     108] = 32'h000000e5;
        data_cache[     109] = 32'h00000065;
        data_cache[     110] = 32'h00000063;
        data_cache[     111] = 32'h00000042;
        data_cache[     112] = 32'h000000a6;
        data_cache[     113] = 32'h00000081;
        data_cache[     114] = 32'h00000078;
        data_cache[     115] = 32'h00000043;
        data_cache[     116] = 32'h00000099;
        data_cache[     117] = 32'h000000d3;
        data_cache[     118] = 32'h000000f4;
        data_cache[     119] = 32'h00000073;
        data_cache[     120] = 32'h00000079;
        data_cache[     121] = 32'h00000096;
        data_cache[     122] = 32'h000000f2;
        data_cache[     123] = 32'h000000eb;
        data_cache[     124] = 32'h000000a5;
        data_cache[     125] = 32'h0000008f;
        data_cache[     126] = 32'h0000004d;
        data_cache[     127] = 32'h00000055;
        data_cache[     128] = 32'h0000002e;
        data_cache[     129] = 32'h00000070;
        data_cache[     130] = 32'h00000022;
        data_cache[     131] = 32'h0000000d;
        data_cache[     132] = 32'h0000009d;
        data_cache[     133] = 32'h000000b2;
        data_cache[     134] = 32'h0000002a;
        data_cache[     135] = 32'h00000039;
        data_cache[     136] = 32'h00000026;
        data_cache[     137] = 32'h000000b1;
        data_cache[     138] = 32'h000000f8;
        data_cache[     139] = 32'h000000ce;
        data_cache[     140] = 32'h00000032;
        data_cache[     141] = 32'h000000ae;
        data_cache[     142] = 32'h0000004e;
        data_cache[     143] = 32'h00000082;
        data_cache[     144] = 32'h000000b0;
        data_cache[     145] = 32'h00000094;
        data_cache[     146] = 32'h0000001f;
        data_cache[     147] = 32'h0000001e;
        data_cache[     148] = 32'h00000054;
        data_cache[     149] = 32'h000000e6;
        data_cache[     150] = 32'h000000cf;
        data_cache[     151] = 32'h00000049;
        data_cache[     152] = 32'h00000011;
        data_cache[     153] = 32'h00000017;
        data_cache[     154] = 32'h00000057;
        data_cache[     155] = 32'h0000006d;
        data_cache[     156] = 32'h000000ad;
        data_cache[     157] = 32'h00000006;
        data_cache[     158] = 32'h000000e1;
        data_cache[     159] = 32'h00000064;
        data_cache[     160] = 32'h0000007b;
        data_cache[     161] = 32'h000000c8;
        data_cache[     162] = 32'h000000f6;
        data_cache[     163] = 32'h00000001;
        data_cache[     164] = 32'h00000044;
        data_cache[     165] = 32'h000000cb;
        data_cache[     166] = 32'h0000007d;
        data_cache[     167] = 32'h000000fd;
        data_cache[     168] = 32'h0000006e;
        data_cache[     169] = 32'h0000008b;
        data_cache[     170] = 32'h000000b6;
        data_cache[     171] = 32'h000000ff;
        data_cache[     172] = 32'h000000df;
        data_cache[     173] = 32'h0000001d;
        data_cache[     174] = 32'h000000de;
        data_cache[     175] = 32'h0000001b;
        data_cache[     176] = 32'h0000005e;
        data_cache[     177] = 32'h0000009a;
        data_cache[     178] = 32'h00000052;
        data_cache[     179] = 32'h000000a9;
        data_cache[     180] = 32'h000000f0;
        data_cache[     181] = 32'h000000f1;
        data_cache[     182] = 32'h000000db;
        data_cache[     183] = 32'h000000e2;
        data_cache[     184] = 32'h00000012;
        data_cache[     185] = 32'h00000080;
        data_cache[     186] = 32'h00000002;
        data_cache[     187] = 32'h00000027;
        data_cache[     188] = 32'h00000058;
        data_cache[     189] = 32'h00000003;
        data_cache[     190] = 32'h0000007a;
        data_cache[     191] = 32'h0000008c;
        data_cache[     192] = 32'h000000c3;
        data_cache[     193] = 32'h0000004c;
        data_cache[     194] = 32'h000000a0;
        data_cache[     195] = 32'h000000e9;
        data_cache[     196] = 32'h00000005;
        data_cache[     197] = 32'h000000fb;
        data_cache[     198] = 32'h0000002f;
        data_cache[     199] = 32'h000000e3;
        data_cache[     200] = 32'h00000007;
        data_cache[     201] = 32'h000000b7;
        data_cache[     202] = 32'h000000fc;
        data_cache[     203] = 32'h00000083;
        data_cache[     204] = 32'h000000b3;
        data_cache[     205] = 32'h00000048;
        data_cache[     206] = 32'h000000d0;
        data_cache[     207] = 32'h0000003f;
        data_cache[     208] = 32'h0000002b;
        data_cache[     209] = 32'h000000e4;
        data_cache[     210] = 32'h00000018;
        data_cache[     211] = 32'h00000056;
        data_cache[     212] = 32'h0000003e;
        data_cache[     213] = 32'h00000088;
        data_cache[     214] = 32'h0000005a;
        data_cache[     215] = 32'h000000bd;
        data_cache[     216] = 32'h00000008;
        data_cache[     217] = 32'h00000030;
        data_cache[     218] = 32'h000000c4;
        data_cache[     219] = 32'h00000059;
        data_cache[     220] = 32'h000000e0;
        data_cache[     221] = 32'h00000047;
        data_cache[     222] = 32'h000000dd;
        data_cache[     223] = 32'h00000077;
        data_cache[     224] = 32'h0000000c;
        data_cache[     225] = 32'h000000af;
        data_cache[     226] = 32'h000000c9;
        data_cache[     227] = 32'h000000ed;
        data_cache[     228] = 32'h00000093;
        data_cache[     229] = 32'h000000a2;
        data_cache[     230] = 32'h00000071;
        data_cache[     231] = 32'h000000d6;
        data_cache[     232] = 32'h00000000;
        data_cache[     233] = 32'h0000006f;
        data_cache[     234] = 32'h000000c1;
        data_cache[     235] = 32'h000000bb;
        data_cache[     236] = 32'h00000098;
        data_cache[     237] = 32'h000000fa;
        data_cache[     238] = 32'h00000023;
        data_cache[     239] = 32'h00000031;
        data_cache[     240] = 32'h00000037;
        data_cache[     241] = 32'h00000090;
        data_cache[     242] = 32'h0000004b;
        data_cache[     243] = 32'h00000087;
        data_cache[     244] = 32'h000000c6;
        data_cache[     245] = 32'h0000005c;
        data_cache[     246] = 32'h0000006a;
        data_cache[     247] = 32'h00000033;
        data_cache[     248] = 32'h0000000f;
        data_cache[     249] = 32'h0000009b;
        data_cache[     250] = 32'h00000050;
        data_cache[     251] = 32'h000000cd;
        data_cache[     252] = 32'h0000008a;
        data_cache[     253] = 32'h00000010;
        data_cache[     254] = 32'h00000084;
        data_cache[     255] = 32'h00000072;
        // ......
    end

    always@(posedge clk)
    begin
        out_data <= addr_valid ? data_cache[dealt_addr] : 32'h0;
        debug_out_data <= debug_addr_valid ? data_cache[dealt_debug_addr] : 32'h0;
        // write data in bytes
        if (write_en[0] & addr_valid)
            data_cache[dealt_addr][7: 0] <= in_data[7:0];
        if (write_en[1] & addr_valid)
            data_cache[dealt_addr][15: 8] <= in_data[15:8];
        if (write_en[2] & addr_valid)
            data_cache[dealt_addr][23:16] <= in_data[23:16];
        if (write_en[3] & addr_valid)
            data_cache[dealt_addr][31:24] <= in_data[31:24];
        // write debug data in bytes
        if (debug_write_en[0] & debug_addr_valid)
            data_cache[dealt_debug_addr][7: 0] <= debug_in_data[7:0];
        if (debug_write_en[1] & debug_addr_valid)
            data_cache[dealt_debug_addr][15: 8] <= debug_in_data[15:8];
        if (debug_write_en[2] & debug_addr_valid)
            data_cache[dealt_debug_addr][23:16] <= debug_in_data[23:16];
        if (debug_write_en[3] & debug_addr_valid)
            data_cache[dealt_debug_addr][31:24] <= debug_in_data[31:24];
    end

endmodule

