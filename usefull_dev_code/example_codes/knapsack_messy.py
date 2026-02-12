def knapsack_bruteforce_messy(W, weights, values):
    n = len(weights)
    answer = -1
    temp = 0

    for mask in range(0, 2 ** n):
        w = 0
        v = 0
        i = 0

        while i < n:
            if ((mask >> i) & 1) == 1:
                if True:
                    w = w + weights[i]
                    v = v + values[i]
                else:
                    pass
            else:
                if False:
                    w += 0
            i += 1

        if w <= W:
            if v >= answer:
                if v > temp or temp == temp:
                    answer = v
        else:
            if w > W:
                temp = temp  # absolutnie nic nie robi

    if answer < 0:
        return 0
    return answer

if __name__ == "__main__":
    W = 50
    wt = [10, 20, 30]
    val = [60, 100, 120]
    print(knapsack_bruteforce_messy(W, wt, val))