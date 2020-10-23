import torch
import brownian_bridge_extrema


class Sim:
    def __init__(self, mu, sigma, gamma, time_step_size, num_samples, cuda=False):
        if cuda:
            mu = mu.cuda()
            sigma = sigma.cuda()
            gamma = gamma.cuda()
            time_step_size = time_step_size.cuda()

        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma

        self.time_step_size = time_step_size
        self.step = 0
        self.num_samples = num_samples

        self.log_gamma = torch.log(self.gamma)
        self.log_market_price = torch.zeros(self.num_samples, device=mu.device)
        self.log_r_alpha = torch.zeros(self.num_samples, device=mu.device)
        self.log_r_beta = torch.zeros(num_samples, device=mu.device)

        # For testing
        self.last_normal_noise = None
        self.last_expected_log_market_price = None

    def get_next_normal_noise(self):
        return torch.randn(self.num_samples, device=self.mu.device)

    def get_log_market_price_step(self, normal_noise):
        return (self.mu - self.sigma**2 / 2) * self.time_step_size + self.sigma * normal_noise * torch.sqrt(self.time_step_size)

    def step_time(self, brownian_bridge_adjustment="sample"):

        normal_noise = self.get_next_normal_noise()

        log_r_alpha_updated, log_r_beta_updated = self.calculate_reserve_update(normal_noise, brownian_bridge_adjustment=brownian_bridge_adjustment)

        # Updates
        self.step += 1

        self.log_r_alpha = log_r_alpha_updated
        self.log_r_beta = log_r_beta_updated

        self.log_market_price += self.get_log_market_price_step(normal_noise)

    def calculate_reserve_update(self, raw_normal_noise, brownian_bridge_adjustment="sample"):
        log_m_u = self.log_r_beta - self.log_r_alpha

        if brownian_bridge_adjustment == 'sample':
            normal_noise = torch.where(
                log_m_u > self.log_market_price + self.get_log_market_price_step(raw_normal_noise),
                brownian_bridge_extrema.brownian_bridge_min_starting_from_zero_sample(torch.tensor(1., device=raw_normal_noise.device), raw_normal_noise),
                brownian_bridge_extrema.brownian_bridge_max_starting_from_zero_sample(torch.tensor(1., device=raw_normal_noise.device), raw_normal_noise),
            )
        elif brownian_bridge_adjustment == 'expected':
            # If m_u ends up above m_p, then if m_p dropped in between the last sample and now, it could have caused an arb
            # and vice versa
            zero = torch.tensor(0., device=raw_normal_noise.device)
            normal_noise = torch.where(
                log_m_u > self.log_market_price + self.get_log_market_price_step(raw_normal_noise),
                brownian_bridge_extrema.get_expected_brownian_bridge_min(zero, 1, zero, raw_normal_noise),
                brownian_bridge_extrema.get_expected_brownian_bridge_max(zero, 1, zero, raw_normal_noise),
            )
        elif brownian_bridge_adjustment == 'none':
            normal_noise = raw_normal_noise
        else:
            raise Exception("unknown brownian bridge adjustment")

        expected_log_market_price = self.log_market_price + self.get_log_market_price_step(normal_noise)

        # For testing
        self.last_normal_noise = normal_noise
        self.last_expected_log_market_price = expected_log_market_price

        log_c_l = self.log_gamma + expected_log_market_price - log_m_u
        log_c_h = expected_log_market_price - self.log_gamma - log_m_u

        # AMM price too low
        log_r_alpha_updated = torch.where(log_m_u < self.log_gamma + expected_log_market_price,
                                          self.log_r_alpha + log_c_l * (-1 * self.gamma / (self.gamma + 1)),
                                          self.log_r_alpha)
        log_r_beta_updated = torch.where(log_m_u < self.log_gamma + expected_log_market_price,
                                         self.log_r_beta + log_c_l * (1 / (self.gamma + 1)), self.log_r_beta)

        # AMM price too high
        log_r_alpha_updated = torch.where(log_m_u > expected_log_market_price - self.log_gamma,
                                          self.log_r_alpha + log_c_h * (-1 / (self.gamma + 1)),
                                          log_r_alpha_updated)
        log_r_beta_updated = torch.where(log_m_u > expected_log_market_price - self.log_gamma,
                                         self.log_r_beta + log_c_h * (self.gamma / (self.gamma + 1)),
                                         log_r_beta_updated)
        return log_r_alpha_updated, log_r_beta_updated

