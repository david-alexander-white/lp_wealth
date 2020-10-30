import numpy as np
from ..lp_wealth import *


def test_log_market_price(cuda=False):
    mu = torch.tensor(3.)
    sigma = torch.tensor(2.)
    time_step_size = torch.tensor(1/100)
    sim = Sim(mu, sigma, torch.tensor(1.), time_step_size, 10000, cuda=cuda)

    steps = 100
    for i in range(steps):
        sim.step_time()

    time = time_step_size * steps

    mean_price = torch.mean(sim.log_market_price)
    expected_mean = (mu - sigma ** 2/ 2) * time
    assert np.abs(mean_price.cpu() - expected_mean) < 0.25

    price_sd = torch.std(sim.log_market_price)
    expected_sd = np.sqrt(time * sigma ** 2)
    assert np.abs(price_sd.cpu() - expected_sd) < 0.25


def test_calculate_reserve_changes(cuda=False):
    gamma = torch.tensor([1, 0.01, 0.5])
    # mu = 1/2 and sigma = 1 so our Brownian motion will get fed
    # directly into our log market price
    sim = Sim(mu=torch.tensor(1/2),
              sigma=torch.tensor(1.),
              gamma=gamma,
              time_step_size=torch.tensor(1.),
              num_samples=3,
              cuda=cuda
              )

    # We're going to pass in a noise realization that will send our log
    # price to exactly where we want it
    fake_noise = torch.log(torch.tensor([1., 1/16, 1/16]))
    if cuda:
        fake_noise = fake_noise.cuda()

    log_r_alpha_low_updated, log_r_beta_low_updated, log_r_alpha_high_updated, log_r_beta_high_updated = \
            sim.calculate_reserve_update(fake_noise, "none")

    # No price change in sample 1, in sample 2 gamma is too high, in sample 3 we finally do something
    assert np.allclose(torch.exp(log_r_alpha_low_updated + log_r_alpha_high_updated).cpu(), torch.tensor([1., 1., 4.]))
    assert np.allclose(torch.exp(log_r_beta_low_updated + log_r_beta_high_updated).cpu(), torch.tensor([1., 1., .5]))

    # Applying our Brownian Bridge adjustment has big effects because our sigma and time step are both big
    # To avoid inequality figdgetyness
    sim.log_market_price += 0.0000001

    check_bb_adjustment(fake_noise, sim, 'expected')
    check_bb_adjustment(fake_noise, sim, 'sample')


def check_bb_adjustment(fake_noise, sim, adjustment_type):
    log_r_alpha_low_updated, log_r_beta_low_updated, log_r_alpha_high_updated, log_r_beta_high_updated = \
        sim.calculate_reserve_update(fake_noise, adjustment_type)
    r_alpha = torch.exp(log_r_alpha_low_updated + log_r_alpha_high_updated)
    r_beta = torch.exp(log_r_beta_low_updated + log_r_beta_high_updated)
    # Sample 1 assumes the price went up between observations (b/c of our fidgetyness adjustment above)
    # So we end up selling some alpha
    assert r_alpha[0] < 1
    assert r_beta[0] > 1
    # Sample 2, price went way down, but gamma still too high
    assert r_alpha[1] == 1
    assert r_beta[1] == 1
    # Sample 3, price goes down more than without the adjustment
    assert r_alpha[2] > 4
    assert r_beta[2] < 0.5

def test_compute_log_wealth():
    num_samples=3
    mu = torch.tensor(num_samples)
    sigma = torch.tensor(num_samples)
    time_step_size = torch.tensor(1 / 100)
    sim = Sim(mu, sigma, torch.tensor(1.), time_step_size, num_samples, cuda=False)

    fake_price = torch.tensor([1.,2,3])
    fake_alpha = torch.tensor([4.,5,6])
    fake_beta = torch.tensor([7.,8,9])

    sim.log_market_price = torch.log(fake_price)
    sim.initial_log_r_alpha = torch.log(fake_alpha)
    sim.initial_log_r_beta = torch.log(fake_beta)

    res = sim.compute_log_wealth()
    assert torch.allclose(torch.exp(res), fake_price * fake_alpha + fake_beta)



if __name__ == "__main__":
    test_compute_log_wealth()
    test_log_market_price(True)
    test_calculate_reserve_changes(True)
