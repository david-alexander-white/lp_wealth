import torch
import numpy as np
import brownian_bridge_extrema


class Sim:
    def __init__(self, mu, sigma, gamma, time_step_size, num_samples):
        self.step = 0

        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma

        self.time_step_size = time_step_size
        self.num_samples = num_samples

        self.log_market_price = torch.zeros(self.num_samples)
        self.log_r_alpha = torch.zeros(self.num_samples)
        self.log_r_beta = torch.zeros(num_samples)

    def get_next_brownian_motion_step(self):
        return torch.randn(self.num_samples) * np.sqrt(self.time_step_size)

    def get_log_market_price_step(self, brownian_motion_step):
        return (self.mu - self.sigma**2 / 2) * self.time_step_size + self.sigma * brownian_motion_step

    def step_time(self):

        brownian_motion_step = self.get_next_brownian_motion_step()

        log_r_alpha_updated, log_r_beta_updated = self.calculate_reserve_update(brownian_motion_step)

        # Updates
        self.step += 1

        self.log_r_alpha = log_r_alpha_updated
        self.log_r_beta = log_r_beta_updated

        self.log_market_price += self.get_log_market_price_step(brownian_motion_step)

    def calculate_reserve_update(self, brownian_motion_step, use_brownian_bridge_adjustment=True):
        log_m_u = self.log_r_beta - self.log_r_alpha

        if use_brownian_bridge_adjustment:
            # If m_u is currently above m_p, then if m_p dropped in between the last sample and now, it could have caused an arb
            # and vice versa
            brownian_motion_step = torch.where(
                log_m_u > self.log_market_price,
                brownian_bridge_extrema.get_expected_brownian_bridge_min(torch.tensor(0.), self.time_step_size, torch.tensor(0.), brownian_motion_step),
                brownian_bridge_extrema.get_expected_brownian_bridge_max(torch.tensor(0.), self.time_step_size, torch.tensor(0.), brownian_motion_step),
            )

        expected_log_market_price = self.log_market_price + self.get_log_market_price_step(brownian_motion_step)

        log_gamma = torch.log(self.gamma)

        log_c_l = log_gamma + expected_log_market_price - log_m_u
        log_c_h = expected_log_market_price - log_gamma - log_m_u

        # AMM price too low
        log_r_alpha_updated = torch.where(log_m_u < log_gamma + expected_log_market_price,
                                          self.log_r_alpha + log_c_l * (-1 * self.gamma / (self.gamma + 1)),
                                          self.log_r_alpha)
        log_r_beta_updated = torch.where(log_m_u < log_gamma + expected_log_market_price,
                                         self.log_r_beta + log_c_l * (1 / (self.gamma + 1)), self.log_r_beta)

        # AMM price too high
        log_r_alpha_updated = torch.where(log_m_u > expected_log_market_price - log_gamma,
                                          self.log_r_alpha + log_c_h * (-1 / (self.gamma + 1)),
                                          log_r_alpha_updated)
        log_r_beta_updated = torch.where(log_m_u > expected_log_market_price - log_gamma,
                                         self.log_r_beta + log_c_h * (self.gamma / (self.gamma + 1)),
                                         log_r_beta_updated)
        return log_r_alpha_updated, log_r_beta_updated

