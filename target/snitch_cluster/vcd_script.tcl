set DUT_VSIM_PATH "tb_bin/i_dut/i_snitch_cluster_netlist"
set VCD_OFFSET $env(VCD_OFFSET)
set VCD_LENGTH $env(VCD_LENGTH)

if { [info exists VCD_OFFSET] == 0 || [info exists VCD_LENGTH] == 0 } {
    error "VCD_OFFSET and VCD_LENGTH must be set"
}

# Initialize the KERNEL_PARAMS list
set KERNEL_PARAMS [list]

# Iterate through all environment variables
foreach var [array names env] {
    # Check if the variable name matches "KERNEL_PARAM_*"
    if {[string match "KERNEL_PARAM_*" $var]} {
        # Remove the "KERNEL_PARAM_" prefix from the key
        set key_no_prefix [string range $var 13 end]
        
        # Append the key (without prefix) and value to the list
        lappend KERNEL_PARAMS "$key_no_prefix=$env($var)"
    }
}

# Display the result
puts "Kernel Params: $KERNEL_PARAMS"

set KERNEL $env(KERNEL)

set FP $env(FP)
set MODE $env(MODE)

set VCD_PATH "snitch-pd/snitch_cluster/target/snitch_cluster/vcd/${KERNEL}_${FP}_${MODE}_${KERNEL_PARAMS}.vcd"

set $VCD_PATH CREAT

set fd [open $VCD_PATH "w"]
close $fd

source /snitch-pd/chip/occamy/gf12/modelsim/occamy_cluster/scripts/init_ff.pls.occamy_cluster.tcl
do wave.do

run $VCD_OFFSET ns
vcd file $VCD_PATH
vcd add -r $DUT_VSIM_PATH/*
run $VCD_LENGTH ns
vcd flush
exit -f