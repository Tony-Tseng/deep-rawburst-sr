from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    # network_list.append(NetworkParam(network_path='dbsr_synthetic_default.pth',
                                    #  unique_name='DBSR_syn'))                   # Evaluate pre-trained network weights
    network_list.append(NetworkParam(module='dbsr', parameter='synthetic_bip'))
#    network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn_diff', epoch=10))
#    network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn_unet_diff', epoch=10))
    # network_list.append(NetworkParam(module='dbsr', parameter='synthetic_dcn'))

    return network_list

