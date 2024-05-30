import numpy as np

def sample_inverse_cost(n_fid, cost_lst, k=1):
    """Samples k elements from list A with probabilities proportional to 1/C.

    Args:
        A: List of items to sample from.
        C: List of corresponding costs.
        k: Number of items to sample (default 1).

    Returns:
        List of sampled items.
    """

    # Calculate inverse costs and normalize for probabilities
    inverse_costs = 1 / np.array(cost_lst)
    probabilities = inverse_costs / inverse_costs.sum()

    # Sample from the list using calculated probabilities
    sampled_indices = np.random.choice(n_fid, size=k, replace=False, p=probabilities)

    return sampled_indices