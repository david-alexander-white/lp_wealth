import torch
import numpy as np
import matplotlib.pyplot as plt


def get_standard_brownian_motion(num_samples, num_steps, step_size):
    w = torch.randn(num_samples, num_steps) * np.sqrt(step_size)
    return torch.cumsum(w, 1)


def get_t(steps, step_size):
    return (torch.arange(0, steps) + 1) * step_size


def get_log_geometric_brownian_motion(z, s_zero, mu, sigma, step_size):
    t = get_t(z.shape[1], step_size)
    return torch.log(s_zero) + (mu - sigma ** 2 / 2) * t + sigma * z


def get_new_log_reserves(log_reserve_tensor, gamma, log_market_price):
    log_r_alpha = log_reserve_tensor[..., 0]
    log_r_beta = log_reserve_tensor[..., 1]

    log_m_u = log_r_beta - log_r_alpha

    log_gamma = torch.log(gamma)

    log_c_l = log_gamma + log_market_price - log_m_u
    log_c_h = log_market_price - log_gamma - log_m_u

    # AMM price too low
    log_r_alpha_updated = torch.where(log_m_u < log_gamma + log_market_price, log_r_alpha + log_c_l * (-1 * gamma / (gamma + 1)),
                                      log_r_alpha)
    log_r_beta_updated = torch.where(log_m_u < log_gamma + log_market_price, log_r_beta + log_c_l * (1 / (gamma + 1)), log_r_beta)

    # AMM price too high
    log_r_alpha_updated = torch.where(log_m_u > log_market_price - log_gamma, log_r_alpha + log_c_h * (-1 / (gamma + 1)),
                                      log_r_alpha_updated)
    log_r_beta_updated = torch.where(log_m_u > log_market_price - log_gamma, log_r_beta + log_c_h * (gamma / (gamma + 1)),
                                     log_r_beta_updated)

    return torch.stack((log_r_alpha_updated, log_r_beta_updated), dim=-1)


def get_log_reserve_tensor(gamma, log_market_price, s_zero):
    out_tensor = torch.zeros(list(log_market_price.shape) + [2])
    out_tensor[..., 1] = torch.log(s_zero)
    for i in range(1, log_market_price.shape[1]):
        if i % max(round(log_market_price.shape[1] / 10), 2) == 1:
            print(i, log_market_price.shape[1])
        out_tensor[:, i, :] = get_new_log_reserves(out_tensor[:, i - 1, :], gamma, log_market_price[:, i])
    return out_tensor
