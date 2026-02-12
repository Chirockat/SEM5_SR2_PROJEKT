def merge_sort(arr):
    # Base case: lista pusta lub 1-elementowa jest posortowana
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    # Rekurencyjne wywołanie dla lewej i prawej strony
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0

    # Główna pętla scalająca
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Dołącz pozostałe elementy (w Pythonie to bardzo szybkie)
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print(merge_sort([4,1,7,4,2,6]))