from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    # network_list.append(NetworkParam(network_path='dbsr_synthetic_default.pth',
    #                                  unique_name='DBSR_syn'))                   # Evaluate pre-trained network weights
    # network_list.append(NetworkParam(module='dbsr', parameter='default_synthetic'))
    
    # Compare Efficient
    network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn_efficient'))
    # network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn_efficient_freeze'))
    # network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn_efficient_freeze_first_2'))
    
    # Compare DCN architecture
    # network_list.append(NetworkParam(module='dcn_arch', parameter='synthetic_2dcn'))
    # network_list.append(NetworkParam(module='dcn_arch', parameter='synthetic_3dcn', epoch=37))
    # network_list.append(NetworkParam(module='dcn_arch', parameter='synthetic_3dcn_NG'))
    # network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn_unet_diff_sch', epoch=85))
    
    # Compare optical Flow or not
    # network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn_unet_diff_sch', epoch=85))
    # network_list.append(NetworkParam(module='dcn_arch', parameter='synthetic_3dcn_NG'))
    # network_list.append(NetworkParam(module='fgdcn', parameter='synthetic_3fgdcn'))
    
    # Compare upsample methods
    # network_list.append(NetworkParam(module='dcn_arch', parameter='synthetic_3dcn_NG', display_name="3dcn"))
    # network_list.append(NetworkParam(module='upsample', parameter='synthetic_3dcn_residual', display_name="3dcn_residual_swin"))
    # network_list.append(NetworkParam(module='upsample', parameter='synthetic_3dcn_residual_noswin', display_name="3dcn_residual"))
    # network_list.append(NetworkParam(module='upsample', parameter='synthetic_3dcn_noresidual', display_name="3dcn_swin"))
    # network_list.append(NetworkParam(module='upsample', parameter='synthetic_3dcn_efficient_residual', display_name="eff_residual"))
    
    # network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn_efficient'))
    # network_list.append(NetworkParam(module='upsample', parameter='synthetic_3dcn_residual'))
    # network_list.append(NetworkParam(module='upsample', parameter='synthetic_3dcn_noresidual'))
    # network_list.append(NetworkParam(module='upsample', parameter='synthetic_3dcn_residual_noswin'))
    

    return network_list

