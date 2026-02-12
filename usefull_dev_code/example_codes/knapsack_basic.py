def knapsack_bruteforce_simple(W, weights, values):
    n = len(weights)
    best_value = 0

    for mask in range(1 << n):
        total_weight = 0
        total_value = 0

        for i in range(n):
            if mask & (1 << i):
                total_weight += weights[i]
                total_value += values[i]

        if total_weight <= W and total_value > best_value:
            best_value = total_value

    return best_value


if __name__ == "__main__":
    W = 50
    wt = [10, 20, 30]
    val = [60, 100, 120]
    print(knapsack_bruteforce_simple(W, wt, val))