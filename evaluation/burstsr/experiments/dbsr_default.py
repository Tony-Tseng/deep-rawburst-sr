from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    # network_list.append(NetworkParam(network_path='dbsr_burstsr_default.pth',
    #                                  unique_name='DBSR_burstsr'))                # Evaluate pre-trained network weights
    # network_list.append(NetworkParam(module='dbsr', parameter='realworld_dcn_unet_diff_sch_w'))
    network_list.append(NetworkParam(module='dbsr', parameter='realworld_dcn_unet_diff_sch', epoch=1))                              

    return network_list

