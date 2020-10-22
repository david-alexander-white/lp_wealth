from lp_wealth import *


def test_log_market_price():
    mu = torch.tensor(3)
    sigma = torch.tensor(2)
    time_step_size = 1/100
    sim = Sim(mu, sigma, torch.tensor(1.), time_step_size, 10000)

    steps = 100
    for i in range(steps):
        sim.step_time()

    time = time_step_size * steps

    mean_price = torch.mean(sim.log_market_price)
    expected_mean = (mu - sigma ** 2/ 2) * time
    assert np.abs(mean_price - expected_mean) < 0.25

    price_sd = torch.std(sim.log_market_price)
    expected_sd = np.sqrt(time * sigma ** 2)
    assert np.abs(price_sd - expected_sd) < 0.25


def test_calculate_reserve_changes():
    gamma = torch.tensor([1, 0.01, 0.5])
    # mu = 1/2 and sigma = 1 so our Brownian motion will get fed
    # directly into our log market price
    sim = Sim(mu=torch.tensor(1/2),
              sigma=torch.tensor(1.),
              gamma=gamma,
              time_step_size=1,
              num_samples=3
              )

    # We're going to pass in a Brownian Motion step that will send our log
    # price to exactly where we want it
    brownian_motion_step = torch.log(torch.tensor([1., 1/16, 1/16]))
    log_r_alpha, log_r_beta = sim.calculate_reserve_update(brownian_motion_step, False)

    # No price change in sample 1, in sample 2 gamma is too high, in sample 3 we finally do something
    assert np.allclose(torch.exp(log_r_alpha), torch.tensor([1.,1.,4.]))
    assert np.allclose(torch.exp(log_r_beta), torch.tensor([1.,1.,.5]))


if __name__ == "__main__":
    test_calculate_reserve_changes()
