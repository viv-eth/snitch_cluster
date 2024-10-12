onerror {resume}
quietly WaveActivateNextPane {} 0

add wave -noupdate -group {Top Level} /tb_bin/rst_ni
add wave -noupdate -group {Top Level} /tb_bin/clk_i
add wave -noupdate -group {Top Level} /tb_bin/i_dut/clk_i
add wave -noupdate -group {Top Level} /tb_bin/i_dut/rst_ni
add wave -noupdate -group {Top Level} /tb_bin/i_dut/narrow_in_req
add wave -noupdate -group {Top Level} /tb_bin/i_dut/narrow_in_resp
add wave -noupdate -group {Top Level} /tb_bin/i_dut/narrow_out_req
add wave -noupdate -group {Top Level} /tb_bin/i_dut/narrow_out_resp
add wave -noupdate -group {Top Level} /tb_bin/i_dut/wide_out_req
add wave -noupdate -group {Top Level} /tb_bin/i_dut/wide_out_resp
add wave -noupdate -group {Top Level} /tb_bin/i_dut/wide_in_req
add wave -noupdate -group {Top Level} /tb_bin/i_dut/wide_in_resp
add wave -noupdate -group {Top Level} /tb_bin/i_dut/msip
add wave -noupdate -group {Top Level} /tb_bin/i_dut/sram_cfgs

add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/clk_i
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/rst_ni_BAR
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/busy_o
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/axi_req_i
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/axi_resp_o
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_r_cnt_q
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_w_cnt_q
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_r_cnt_d
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_w_cnt_d
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_sel_buf_usage_o
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_sel_buf_fifo_i_read_pointer_q
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_sel_buf_fifo_i_mem_q
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_meta_buf_usage_o
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_meta_buf_fifo_i_read_pointer_q
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_meta_buf_fifo_i_write_pointer_q
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_stream_to_mem_cnt_d
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_stream_to_mem_cnt_q
add wave -noupdate -group {AXI} /tb_bin/i_dut/i_snitch_cluster_netlist/i_cluster_i_axi_zeromem/i_axi_to_zeromem_i_axi_to_detailed_mem_i_stream_to_mem_gen_buf_i_resp_buf_usage_o

