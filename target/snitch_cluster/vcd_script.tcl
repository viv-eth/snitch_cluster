set DUT_VSIM_PATH "tb_bin/i_dut/i_snitch_cluster_netlist"
set VCD_OFFSET 37939
set VCD_LENGTH 17601

if { [info exists VCD_OFFSET] == 0 || [info exists VCD_LENGTH] == 0 } {
    error "VCD_OFFSET and VCD_LENGTH must be set"
}

set MAT_M 48
set MAT_N 48
set MAT_K 1536

set M 64
set N 32
set M_TILES 2

set ALPHA 1.5
set BETA 3.2

set R 16
set Q 16
set S 32
set R_TILES 2
set Q_TILES 2

set KERNEL "GEMM"

set FP "8"
set MODE "OPT_EX"

set VCD_PATH "/scratch/vivianep/snitch-pd/snitch_cluster/target/snitch_cluster/vcd/${KERNEL}_${MAT_M}x${MAT_N}x${MAT_K}_FP${FP}_${MODE}_V2.vcd"
# set VCD_PATH "/scratch/vivianep/snitch-pd/snitch_cluster/target/snitch_cluster/vcd/${KERNEL}_${MODE}_${M}x${N}_${M_TILES}.vcd"
# set VCD_PATH "/scratch/vivianep/snitch-pd/snitch_cluster/target/snitch_cluster/vcd/${KERNEL}_${MODE}_${R}x${Q}x${S}_${R_TILES}_${Q_TILES}.vcd"

set $VCD_PATH CREAT

set fd [open $VCD_PATH "w"]
close $fd

source /usr/scratch2/patagonia/vivianep/snitch-pd/chip/occamy/gf12/modelsim/occamy_cluster/scripts/init_ff.pls.occamy_cluster.tcl
do wave.do

run $VCD_OFFSET ns
vcd file $VCD_PATH
vcd add -r $DUT_VSIM_PATH/*
run $VCD_LENGTH ns
vcd flush
exit -f