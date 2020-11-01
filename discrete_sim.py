from lp_wealth import Sim
import torch

class DiscreteSim(Sim):
    def __init__(self, p, delta, gamma, num_samples, cuda=False):
        super().__init__(torch.tensor(0), torch.tensor(0), gamma, torch.tensor(1.), num_samples, cuda)
        self.p = p
        self.delta = delta

    def get_next_noise(self, sample_style='multi'):
        if sample_style == 'multi':
            inp = torch.ones(self.num_samples, device = self.mu.device, dtype=self.mu.dtype) * self.p
        elif sample_style == 'single':
            inp = torch.tensor(self.p, device=self.mu.device, dtype=self.mu.dtype)
        else:
            return "unknown sample style"

        bernoulli = torch.bernoulli(inp)

        return bernoulli * 2 - 1

    def get_log_market_price_step(self, noise):
        return self.delta * noise
