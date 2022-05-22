from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    # network_list.append(NetworkParam(network_path='dbsr_burstsr_default.pth',
    #                                  unique_name='DBSR_burstsr'))                # Evaluate pre-trained network weights
    network_list.append(NetworkParam(module='dbsr', parameter='default_realworld'))
    network_list.append(NetworkParam(module='dbsr', parameter='default_realworld_SSL'))
    network_list.append(NetworkParam(module='dbsr', parameter='default_realworld_SSLRAW'))

    return network_list