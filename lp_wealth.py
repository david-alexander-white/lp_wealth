import torch
import brownian_bridge_extrema


class Sim:
    def __init__(self, mu, sigma, gamma, time_step_size, num_samples, cuda=False):
        # Safety
        mu = mu.float()
        sigma = sigma.float()
        gamma = gamma.float()
        time_step_size = time_step_size.float()

        # Cuda
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

        # Log market price 0 ==> market price 1
        self.log_market_price = torch.zeros(self.num_samples, device=mu.device, dtype=mu.dtype)

        # We initialize our reserves so that our initial wealth is 1
        self.initial_log_r_alpha = torch.log(.5 * torch.ones(self.num_samples, device=mu.device, dtype=mu.dtype))
        self.initial_log_r_beta = torch.log(.5 * torch.ones(num_samples, device=mu.device, dtype=mu.dtype))

        # Updates
        self.log_r_alpha_high_updates = torch.zeros(self.num_samples, device=mu.device, dtype=mu.dtype)
        self.log_r_alpha_low_updates = torch.zeros(self.num_samples, device=mu.device, dtype=mu.dtype)

        self.log_r_beta_high_updates = torch.zeros(self.num_samples, device=mu.device, dtype=mu.dtype)
        self.log_r_beta_low_updates = torch.zeros(self.num_samples, device=mu.device, dtype=mu.dtype)

        self.trade_count = torch.zeros(self.num_samples, device=mu.device, dtype=mu.dtype)

        # For testing
        self.last_normal_noise = None
        self.last_expected_log_market_price = None

        self.last_r_alpha = torch.exp(self.initial_log_r_alpha)
        self.last_r_beta = torch.exp(self.initial_log_r_beta)

    def get_log_r_alpha(self):
        return self.initial_log_r_alpha + self.log_r_alpha_high_updates + self.log_r_alpha_low_updates

    def get_log_r_beta(self):
        return self.initial_log_r_beta + self.log_r_beta_high_updates + self.log_r_beta_low_updates

    def get_r_alpha(self):
        return torch.exp(self.get_log_r_alpha())

    def get_r_beta(self):
        return torch.exp(self.get_log_r_beta())

    def get_next_noise(self, sample_style='multi'):
        if sample_style == 'multi':
            return torch.randn(self.num_samples, device=self.mu.device, dtype=self.mu.dtype)
        elif sample_style == 'single':
            return torch.randn(1, device=self.mu.device, dtype=self.mu.dtype)
        else:
            return "unknown sample style"

    def get_log_market_price_step(self, noise):
        return (self.mu - self.sigma**2 / 2) * self.time_step_size + self.sigma * noise * torch.sqrt(self.time_step_size)

    def step_time(self, brownian_bridge_adjustment="sample", sample_style='multi', provided_noise=None):
        if provided_noise:
            assert brownian_bridge_adjustment == 'none' and sample_style == 'provided'

        self.last_r_alpha = self.get_r_alpha()
        self.last_r_beta = self.get_r_beta()

        with torch.no_grad():
            noise = self.get_next_noise(sample_style) if provided_noise is None else provided_noise

            log_r_alpha_low_updated, log_r_beta_low_updated, log_r_alpha_high_updated, log_r_beta_high_updated \
                = self.calculate_reserve_update(noise, brownian_bridge_adjustment=brownian_bridge_adjustment, sample_style=sample_style)

            # Updates
            self.step += 1

            self.log_market_price += self.get_log_market_price_step(noise)

            self.trade_count += torch.max(self.log_r_alpha_high_updates != log_r_alpha_high_updated,
                                self.log_r_alpha_low_updates != log_r_alpha_low_updated)

            self.log_r_alpha_low_updates = log_r_alpha_low_updated
            self.log_r_alpha_high_updates = log_r_alpha_high_updated

            self.log_r_beta_low_updates = log_r_beta_low_updated
            self.log_r_beta_high_updates = log_r_beta_high_updated

            return

    def calculate_reserve_update(self, raw_normal_noise, brownian_bridge_adjustment="sample", sample_style='multi'):
        log_m_u = self.get_log_r_beta() - self.get_log_r_alpha()

        if brownian_bridge_adjustment == 'sample':
            normal_noise = torch.where(
                log_m_u > self.log_market_price + self.get_log_market_price_step(raw_normal_noise),
                brownian_bridge_extrema.brownian_bridge_min_starting_from_zero_sample(torch.tensor(1., device=raw_normal_noise.device), raw_normal_noise, sample_style=sample_style),
                brownian_bridge_extrema.brownian_bridge_max_starting_from_zero_sample(torch.tensor(1., device=raw_normal_noise.device), raw_normal_noise, sample_style=sample_style),
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
        log_r_alpha_low_updated = torch.where(log_m_u < self.log_gamma + expected_log_market_price,
                                          self.log_r_alpha_low_updates + log_c_l * (-1 * self.gamma / (self.gamma + 1)),
                                          self.log_r_alpha_low_updates)
        log_r_beta_low_updated = torch.where(log_m_u < self.log_gamma + expected_log_market_price,
                                         self.log_r_beta_low_updates + log_c_l * (1 / (self.gamma + 1)), self.log_r_beta_low_updates)

        # AMM price too high
        log_r_alpha_high_updated = torch.where(log_m_u > expected_log_market_price - self.log_gamma,
                                          self.log_r_alpha_high_updates + log_c_h * (-1 / (self.gamma + 1)),
                                          self.log_r_alpha_high_updates)
        log_r_beta_high_updated = torch.where(log_m_u > expected_log_market_price - self.log_gamma,
                                         self.log_r_beta_high_updates + log_c_h * (self.gamma / (self.gamma + 1)),
                                         self.log_r_beta_high_updates)
        return log_r_alpha_low_updated, log_r_beta_low_updated, log_r_alpha_high_updated, log_r_beta_high_updated

    # Compute log(r_alpha * S + r_beta)
    # We go through a few contortions to avoid scaling problems
    def compute_log_wealth(self):
        log_alpha_value = self.log_market_price + self.get_log_r_alpha()

        # Divide through by value of beta (which represents cash)
        scaled_log_alpha_value = log_alpha_value - self.get_log_r_beta()

        scaled_wealth = torch.exp(scaled_log_alpha_value) + 1

        scaled_log_wealth = torch.log(scaled_wealth)

        return (scaled_log_wealth + self.get_log_r_beta()).cpu()

    def compute_log_amm_price(self):
        return (self.get_log_r_beta() - self.get_log_r_alpha()).cpu()

    def compute_wealth(self):
        return torch.exp(self.compute_log_wealth())

    def compute_elapsed_time(self):
        return (self.step * self.time_step_size).cpu()

    def compute_wealth_growth_rate(self):
        return self.compute_log_wealth() / self.compute_elapsed_time()

    # See https://math.dartmouth.edu/~mtassy/articles/AMM_returns.pdf
    def predict_wealth_growth_rate(self):
        d = self.mu - self.sigma ** 2 / 2
        return d/2 * ((1 + self.gamma**(4 * d/self.sigma**2))/(1 - self.gamma**(4 * d/self.sigma**2))*(1-self.gamma)/(1+self.gamma)+1)

    def compute_trade_rate(self):
        return self.trade_count.cpu() / self.compute_elapsed_time()

