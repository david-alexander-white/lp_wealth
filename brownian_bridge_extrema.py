import torch, numpy as np

def get_expected_brownian_bridge_max_starting_from_0_and_ending_at_or_above_0(end_time, end_point):
    # This is derived from https://www.researchgate.net/publication/236984395_On_the_maximum_of_the_generalized_Brownian_bridge eq. (2.7)
    # to get the PDF and then integrating to get EV
    return end_point + \
           1/2 * torch.sqrt(np.pi / 2 * end_time) * torch.exp(end_point ** 2 / (2 * end_time)) \
           * torch.erfc(end_point / torch.sqrt(2 * end_time))

def get_expected_brownian_bridge_max_starting_from_0(end_time, end_point):
    # If end_point is ever negative, we're going to shift all paths up by end_point units and then reflect them
    # horizontally, so that we now our paths start at 0 and end at abs(end_point)
    # Conveniently enough, we can do this just by switching end_point to abs(end_point)
    positive_endpoint = torch.abs(end_point)
    positive_endpoint_max = get_expected_brownian_bridge_max_starting_from_0_and_ending_at_or_above_0(end_time, positive_endpoint)

    # If we reflected anybody, we need to offset their calculated max (by shifting the new end point back down to 0)
    return torch.where(end_point < 0, positive_endpoint_max + end_point, positive_endpoint_max)

def get_expected_brownian_bridge_max(start_time, end_time, start_point, end_point):
    time_elapsed = end_time - start_time
    distance_traveled = end_point - start_point

    max_from_zero = get_expected_brownian_bridge_max_starting_from_0(time_elapsed, distance_traveled)

    return max_from_zero + start_point

#############
# Min
#############
def get_expected_brownian_bridge_min(start_time, end_time, start_point, end_point):
    # Any path will be at its minimum when the negative of that path is at its maximum, meaning
    # our answer is the negative max of our negative
    return -1 * get_expected_brownian_bridge_max(start_time, end_time, -start_point, -end_point)
