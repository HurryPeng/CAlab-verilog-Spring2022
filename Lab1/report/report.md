# Verilog-lab1 实验报告

彭浩然 PB19051055

## 设计修改

1. 将各个级间寄存器的`Bubble`和`Flush`控制信号合成一个`Pipeline`控制信号，用常量`PASS`、`HOLD`和`FLUSH`分别表示下一周期读入上一级的数据、保持本级数据和清零本级数据，这样更加明确每个级间寄存器每个周期只能执行一种动作，可以避免对`Bubble`和`Flush`两个信号优先级的困惑，减少代码中线路的数量。
2. 将随流水线传递下去的PC + 4改成传递PC，这样可以避免ID级和EX级需要PC时插入两个`-4`模块，在`ALU MUX`处的`PCE`信号真正需要PC + 4的时候再加入`+4`模块即可。一方面，这把两个减法器改为了一个加法器，节省了器件；另一方面，从性能考虑，需要经过ALU的EX级常常成为关键路径，在ALU执行前先执行一个减法简直是雪上加霜，而把`-4`去掉换成`EPC`前的`+4`可以平均各路径时延，让`ALU MUX`的两个输入信号都只经过一个算术运算。

## 问题回答

> 1. 描述执行一条 XOR 指令的过程（数据通路、控制信号等）

1-3题的高清大图请见压缩包

![PART-RV32I-Core-Design-Figure-XOR](./assets/PART-RV32I-Core-Design-Figure-XOR.png)

> 2. 描述执行一条 BEQ 指令的过程·（数据通路、控制信号等）

![PART-RV32I-Core-Design-Figure-BEQ](./assets/PART-RV32I-Core-Design-Figure-BEQ.png)

> 3. 描述执行一条 LHU 指令的过程（数据通路、控制信号等）

![PART-RV32I-Core-Design-Figure-LHU](./assets/PART-RV32I-Core-Design-Figure-LHU.png)

> 4. 如果要实现 CSR 指令（csrrw，csrrs，csrrc，csrrwi，csrrsi，csrrci），设计图中还需要增加什么部件和数据通路？给出详细说明。 

CSR寄存器采用一个单读单写寄存器堆实现，在ID级进行读，WB段进行写。其读地址来自于立即数；读取结果需要增加一系列级间寄存器传递至MEM级，在EX级接入Op2 MUX，在MEM级接入WB MUX；写地址（即立即数域）需要增加一条与通用寄存器写回地址类似的数据通路，从ID级一直传递到WB级使用；写数据从MEM级的ALU Result引出，经过一级WB寄存器后写回。

一条CSR指令的执行流程是，IF级取指后，ID级同时取出CSR相应寄存器的值以及rs1的值，EX级中ALU将旧CSR值与rs1的值按照CSR指令类型进行覆盖、置位或清零位运算（需要ALU功能以及ALU输入选择MUX的额外支持），得到新CSR值，MEM级不做操作，仅仅是传递旧CSR值和新CSR值，最后在WB段向相应CSR寄存器写回新CSR值，向rd写回旧CSR值。

> 5. Verilog 如何实现立即数的扩展？ 

可以使用连接符`{}`和连续赋值语句组合实现立即数扩展。零扩展只需重复0即可，符号位扩展需要重复最高位。例如：

```verilog
assign immSext = { { 20{ inst[31] } }, inst[31:20] };
assign immZext = { 20'b0, inst[31:20] };
```

> 6. 如何实现 Data Memory 的非字对齐的 Load 和 Store？ 

需要在Data Memory（或者Cache）中加入一个状态机和一个临时寄存器，用两个周期完成非对齐访问。当检测到访存地址非对齐时，需要将流水线先暂停，用第一个周期取部分数据暂存在临时寄存器里（或从临时寄存器里写入部分数据），第二个周期取剩下的数据并和临时寄存器里的数据一起输出（或写入剩下的数据）。

> 7. ALU 模块中，默认 wire 变量是有符号数还是无符号数？ 

无符号数，Verilog中所有整型变量都默认为无符号数。

> 8. 简述BranchE信号的作用。

（图上没找到BranchE信号，我解释BR信号的作用）

BR信号在EX级执行时给出是否进行跳转的判断，PC收到这个信号后会将下一条指令写为跳转目标地址，Hazard Unit收到这个信号后会flush掉EX级前的级间寄存器。

> 9. NPC Generator 中对于不同跳转 target 的选择有没有优先级？ 

有优先级。因为BR和JALR信号是从EX段传来的，而JAL信号是从ID段传来的，所以当它们之中的两个同时出现时，BR或JALR是逻辑上先发射的指令，后方的JAL是否执行要取决于BR和JALR是否跳转。因此，BR和JALR优先级高（这两个不可能同时出现），而JAL优先级低。

> 10. Harzard 模块中，有哪几类冲突需要插入气泡，分别使流水线停顿几个周期？ 

只有Load-Use型冲突需要插入气泡，使流水线停顿一个周期。跳转相关的flush另外处理。

> 11. Harzard 模块中采用静态分支预测器，即默认不跳转，遇到 branch指令时，如何控制 flush 和 stall 信号？ 

遇到经过EX级判断确认跳转的branch指令时，BR信号为1，需要flush掉EX级和ID级寄存器（即`PipelineE`和`PipelineD`控制信号设为`FLUSH`）。IF级寄存器（也就是PC）会根据BR信号自行更改，不需要Hazard Unit处理。

> 12. 0 号寄存器值始终为 0，是否会对 forward 的处理产生影响？

会，如果先发射的某条指令a写回地址为`x0`（即计算结果不写回），而后发射的一条指令b的一个操作数恰为`x0`（即取0），那么前递模块可能会错误地将指令a本该被抛弃的计算结果前递给指令b。解决这个问题的方法是，将操作数为`x0`的前递值硬编码为0。

