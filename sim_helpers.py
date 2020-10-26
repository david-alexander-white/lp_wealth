import time
import torch

def sim_loop(sim, num_steps, bb_adjustment_type="sample", sample_style="multi"):
    checkpoints = []
    start = time.time()
    for i in range(num_steps):
        if i % int(num_steps / 10) == 1:
            now = time.time()
            print(i, (now - start))
            start = now
            checkpoints.append(torch.clone(sim.compute_wealth_growth_rate()))
        sim.step_time(bb_adjustment_type, sample_style=sample_style)
    return checkpoints


def get_max_time_step(gamma, sigma):
    return torch.log(gamma) **2 / 4 / sigma **2


def get_min_time(sigma, gamma):
    return 400 * (torch.log(gamma) / sigma)**2


def get_min_time_steps(sigma, gamma, time_step):
    min_tensor = get_min_time(sigma, gamma) / time_step
    return int(torch.max(min_tensor))