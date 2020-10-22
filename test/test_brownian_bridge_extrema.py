from brownian_bridge_extrema import *

def test_get_expected_brownian_bridge_max():
    # Real answers derived using the sde package in R

    # > a <- 0; mean(sapply(1:10000, function(x) max(BBridge(0,a,0,1,1000))))
    # [1] 0.6065004
    ans = get_expected_brownian_bridge_max_starting_from_0_and_ending_at_or_above_0(torch.tensor(1.), torch.tensor(0.))

    assert torch.abs(ans - 0.6065) < 0.03

    #> a <- 1; mean(sapply(1:100000, function(x) max(BBridge(0,a,0,1,1000))))
    #[1] 1.308907
    ans = get_expected_brownian_bridge_max_starting_from_0_and_ending_at_or_above_0(torch.tensor(1.), torch.tensor(1.))

    assert torch.abs(ans - 1.30809) < 0.03

    # > a <- -1; mean(sapply(1:100000, function(x) max(BBridge(0,a,0,1,1000))))
    # [1] 0.3116
    ans = get_expected_brownian_bridge_max_starting_from_0(torch.tensor(1.), torch.tensor(-1.))

    assert torch.abs(ans - 0.3116) < 0.03

    # > a <- 2;  b <- 1; mean(sapply(1:10000, function(x) max(BBridge(a,b,2,4,1000))))
    # [1] 2.513275
    ans = get_expected_brownian_bridge_max(torch.tensor(2.), torch.tensor(4.), torch.tensor(2.),
                                           torch.tensor(1.))

    assert torch.abs(ans - 2.513) < 0.04

def test_get_expected_brownian_bridge_min():
    # > a <- 2;  b <- 1; mean(sapply(1:10000, function(x) min(BBridge(a,b,2,4,1000))))
    # [1] 0.47752
    ans = get_expected_brownian_bridge_min(torch.tensor(2.), torch.tensor(4.), torch.tensor(2.),
                                           torch.tensor(1.))
    assert torch.abs(ans - 0.47752) < 0.04

    #> a <- -1;  b <- -2; mean(sapply(1:10000, function(x) min(BBridge(a,b,3,4,1000))))
    # [1] -2.307527
    ans = get_expected_brownian_bridge_min(torch.tensor(3.), torch.tensor(4.), torch.tensor(-1.),
                                           torch.tensor(-2.))
    assert torch.abs(ans + 2.307) < 0.04



if __name__ == "__main__":
    test_get_expected_brownian_bridge_max()
    test_get_expected_brownian_bridge_min()

