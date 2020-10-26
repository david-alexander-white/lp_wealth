import lp_wealth
from sim_helpers import *


def test():
    num_samples = 5

    # sigma = torch.rand(num_samples) * 5
    sigma = torch.tensor([0.25, 0.5, 1.,2.,4.])
    mu = sigma ** 2 / 2
    gamma = torch.exp(-sigma)

    time_step = torch.min(get_max_time_step(gamma, sigma))
    num_steps = get_min_time_steps(sigma, gamma, time_step)

    sim = lp_wealth.Sim(mu, sigma, gamma, time_step, num_samples, cuda=True)
    # sim_sigma = lp_wealth.Sim(torch.tensor(10), sigma, gamma, time_step, num_samples, cuda=True)
    # sim_both = lp_wealth.Sim(mu, sigma, gamma, time_step, num_samples, cuda=True)

    sim_loop(sim, num_steps, bb_adjustment_type='none', sample_style='single')


if __name__ == "__main__":
    test()