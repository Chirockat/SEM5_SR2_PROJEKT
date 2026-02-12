def messy_merge_sort(arr):
    # Sprawdzamy czy w ogóle jest co sortować
    if len(arr) > 1:
        mid = len(arr) // 2

        # Tworzenie tymczasowych list "na piechotę"
        L = arr[:mid]
        R = arr[mid:]

        messy_merge_sort(L)
        messy_merge_sort(R)

        i = 0
        j = 0
        k = 0

        # Logika scalania upchnięta w głównej funkcji
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i = i + 1
            else:
                arr[k] = R[j]
                j = j + 1
            k = k + 1

        # Dopinanie resztek ręcznymi pętlami
        while i < len(L):
            arr[k] = L[i]
            i = i + 1
            k = k + 1

        while j < len(R):
            arr[k] = R[j]
            j = j + 1
            k = k + 1

    return arr

print(messy_merge_sort([4,1,7,4,2,6]))