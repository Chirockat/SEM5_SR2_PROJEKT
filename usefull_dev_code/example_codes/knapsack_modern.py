from itertools import product

def knapsack_bruteforce_modern(W, weights, values):
    n = len(weights)
    return max(
        sum(v for i, v in enumerate(values) if mask[i])
        for mask in product([0, 1], repeat=n)
        if sum(w for i, w in enumerate(weights) if mask[i]) <= W
    )

if __name__ == "__main__":
    W = 50
    wt = [10, 20, 30]
    val = [60, 100, 120]
    print(knapsack_bruteforce_modern(W, wt, val))