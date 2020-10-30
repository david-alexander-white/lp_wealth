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


def get_max_time_step(gamma, sigma, mu):
    # We want to make sure trading back and forth in one timestep is a 4-sigma event
    sigma_timestep = torch.log(gamma) **2 / 4 / sigma **2

    # We want to make sure that the drift in any given timestep is at most 1/5 sigma (which scales with sqrt(t))
    # If we have timesteps were std dev is comparable to drift, we should expect to miss out on a lot of trades
    # in the opposite direction from drift.
    mu_timestep = sigma ** 2 / 25 / (mu-sigma**2/2)**2 if mu is not None else torch.tensor(1.e6)

    return torch.min(sigma_timestep, mu_timestep)


def get_min_time(sigma, gamma):
    # The standard deviation of log m_p at time t is sigma * sqrt(t)
    # If that's 10 * log(gamma), then we should expect to see at least a couple trades
    return (10 * torch.log(gamma) / sigma)**2


def get_min_time_steps(sigma, gamma, time_step):
    min_tensor = get_min_time(sigma, gamma) / time_step
    return int(torch.max(min_tensor))