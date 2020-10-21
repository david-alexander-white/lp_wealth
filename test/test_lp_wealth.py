from lp_wealth import *


def test_standard_brownian_motion():
    motion = get_standard_brownian_motion(10000, 100, 0.02)

    mean = torch.mean(motion[:, -1])
    assert np.abs(mean - 0) < 0.1

    var = torch.var(motion[:, -1])
    assert np.abs(var - 2) < 0.1


def test_get_log_reserve_tensor():
    # Market price tensor (3 samples, 2 time steps)
    s = torch.ones(3, 2)

    # Price of alpha at t1
    s[0, 1] = 1
    s[1, 1] = 1/16
    s[2, 1] = 1/16

    gamma = torch.tensor([1, 0.01, 0.5])
    log_price = torch.log(s)
    s_zero = torch.tensor(1.)
    res = torch.exp(get_log_reserve_tensor(gamma, log_price, s_zero))[:, -1, :]

    # No price change for sample 1
    assert np.allclose(res[0, :], [1, 1])
    # Sample 2 has a price change but gamma is too high
    assert np.allclose(res[1, :], [1, 1])
    # Sample has price drop, gamma is 0.5, we end up buying 3 of coin alpha
    assert np.allclose(res[2, :], [4, .5])


if __name__ == "__main__":
    test_standard_brownian_motion()
    test_get_log_reserve_tensor()