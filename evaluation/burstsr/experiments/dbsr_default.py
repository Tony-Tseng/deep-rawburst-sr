from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    # network_list.append(NetworkParam(network_path='dbsr_burstsr_default.pth',
    #                                  unique_name='DBSR_burstsr'))                # Evaluate pre-trained network weights
    # network_list.append(NetworkParam(unique_name='DBSR_results'))
    # network_list.append(NetworkParam(network_path='/home/tony/Desktop/deep-rawburst-sr/exp/checkpoints/dbsr/default_synthetic/DBSRNet_ep0100.pth.tar', unique_name='DBSR_syn'))
    network_list.append(NetworkParam(module='dbsr', parameter='default_realworld'))
    # network_list.append(NetworkParam(module='dbsr', parameter='default_realworld_SSL_tune'))
    # network_list.append(NetworkParam(module='dbsr', parameter='default_realworld_SSLRAW'))
    # network_list.append(NetworkParam(module='dbsr', parameter='realworld_SSL'))
    # network_list.append(NetworkParam(module='dbsr', parameter='realworld_SSL_kernel'))
    # network_list.append(NetworkParam(module='dbsr', parameter='realworld_SSL_tune'))

    return network_list