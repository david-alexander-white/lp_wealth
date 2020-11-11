from lp_wealth import *

def test_provided():
    mu = torch.tensor(0.)
    sigma = torch.tensor(1.)
    gamma = torch.tensor([0.75, 1])
    time_step_size = torch.tensor(1.)
    num_samples = 2

    sim = Sim(mu, sigma, gamma, time_step_size, num_samples)

    while True:
        sim.step_time('none', 'provided', torch.tensor(1.) + sigma**2/2)
        sim.step_time('none', 'provided', torch.tensor(-1.) + sigma**2/2)

if __name__ == "__main__":
    test_provided()
