from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    # network_list.append(NetworkParam(network_path='dbsr_synthetic_default.pth',
    #                                  unique_name='DBSR_syn'))                   # Evaluate pre-trained network weights
    network_list.append(NetworkParam(module='dbsr', parameter='default_synthetic'))
    # network_list.append(NetworkParam(module='dbsr', parameter='synthetic_degenerate'))
    network_list.append(NetworkParam(module='dbsr', parameter='synthetic_SSL'))
    # network_list.append(NetworkParam(module='dbsr', parameter='synthetic_BYOL'))
    network_list.append(NetworkParam(module='dbsr', parameter='synthetic_SSL_1way'))
    network_list.append(NetworkParam(module='dbsr', parameter='synthetic_SSLRAW'))
    # network_list.append(NetworkParam(module='dbsr', parameter='default_synthetic_pcl'))

    return network_list