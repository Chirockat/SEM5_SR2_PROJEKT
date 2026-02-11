import os
import shutil
import sys

# Próbujemy znaleźć bibliotekę
try:
    import pyit2fls
except ImportError:
    print("BŁĄD: Nie znaleziono biblioteki pyit2fls.")
    sys.exit(1)

# --- NOWY KOD FUNKCJI EIASC ---
NEW_EIASC_CODE = r'''def EIASC_algorithm(intervals, params=[]):
    """
    EIASC algorithm (Poprawiona wersja - v3)
    """

    # Left calculations
    intervals = trim(intervals)

    if intervals is False:
        return 0., 0.

    N = len(intervals)

    # FIX 1: Obsługa pojedynczej reguły
    if N == 1:
        return intervals[0, 0], intervals[0, 1]

    intervals = intervals[intervals[:, 0].argsort()]
    a_l = npsum(intervals[:, 0] * intervals[:, 2])
    b_l = npsum(intervals[:, 2])
    L = 0

    # FIX 2: Bezpieczna pętla while
    while L < N:
        d = intervals[L, 3] - intervals[L, 2]
        a_l += intervals[L, 0] * d
        b_l += d

        if b_l == 0:
            y_l = intervals[L, 0]
        else:
            y_l = a_l / b_l

        if (y_l <= intervals[L, 0]) or isclose(y_l, intervals[L, 0]):
            break
        L += 1

    # Right calculations
    intervals = intervals[intervals[:, 1].argsort()]
    a_r = npsum(intervals[:, 1] * intervals[:, 2])
    b_r = npsum(intervals[:, 2])
    R = N - 1

    # FIX 3: Bezpieczna pętla while (prawa strona)
    while R >= 0:
        d = intervals[R, 3] - intervals[R, 2]
        a_r += intervals[R, 1] * d
        b_r += d

        if b_r == 0:
            y_r = intervals[R, 1]
        else:
            y_r = a_r / b_r

        if (y_r >= intervals[R, 1]) or isclose(y_r, intervals[R, 1]):
            break
        R -= 1

    return y_l, y_r


'''


def patch_library_absolute():
    # 1. Ustalanie właściwej ścieżki do pliku z kodem (nie __init__.py)
    package_dir = os.path.dirname(pyit2fls.__file__)
    target_file = os.path.join(package_dir, "pyit2fls.py")

    print(f"[INFO] Folder pakietu: {package_dir}")
    print(f"[INFO] Celujemy w plik: {target_file}")

    if not os.path.exists(target_file):
        print("[BŁĄD] Nie znaleziono pliku pyit2fls.py w folderze pakietu!")
        return

    # 2. Backup
    if not os.path.exists(target_file + ".backup_v3"):
        shutil.copy(target_file, target_file + ".backup_v3")
        print(f"[INFO] Backup utworzony: {target_file}.backup_v3")

    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- KROK 1: Podmiana EIASC ---

    # Szukamy nagłówków funkcji dokładnie tak, jak są w pliku
    start_marker = "def EIASC_algorithm(intervals, params=[]):"
    # Szukamy następnej funkcji, żeby wiedzieć gdzie skończyć wycinanie
    end_marker = "def WM_algorithm(intervals, params=[]):"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        print(f"[INFO] Znaleziono funkcję EIASC (index: {start_idx}). Podmieniam...")
        # Sklejamy nową treść
        content = content[:start_idx] + NEW_EIASC_CODE + content[end_idx:]
    elif "EIASC algorithm (Poprawiona wersja - v3)" in content:
        print("[INFO] Kod EIASC jest już zaktualizowany.")
    else:
        print("[BŁĄD] Nie udało się dopasować sygnatur funkcji w pliku pyit2fls.py.")
        print("Upewnij się, że plik nie był modyfikowany ręcznie w niestandardowy sposób.")
        # Ratunkowe szukanie bez params=[]
        print("Próbuję trybu awaryjnego...")
        start_idx = content.find("def EIASC_algorithm(")
        end_idx = content.find("def WM_algorithm(")
        if start_idx != -1 and end_idx != -1:
            content = content[:start_idx] + NEW_EIASC_CODE + content[end_idx:]
            print("[INFO] Udało się w trybie awaryjnym!")
        else:
            return

    # --- KROK 2: Naprawa SyntaxWarning ---
    print("[INFO] Naprawiam SyntaxWarning...")

    # Celujemy w specyficzne zakończenia definicji funkcji, które wkleiłeś
    # T1FS_plot
    bad_syntax_t1 = 'bbox_to_anchor=None, alpha=0.5, ):\n    """'
    good_syntax_t1 = 'bbox_to_anchor=None, alpha=0.5, ):\n    r"""'

    # IT2FS_plot
    bad_syntax_it2 = 'bbox_to_anchor=None, alpha=0.5, ):\n    """'
    good_syntax_it2 = 'bbox_to_anchor=None, alpha=0.5, ):\n    r"""'

    count = 0
    if bad_syntax_t1 in content:
        content = content.replace(bad_syntax_t1, good_syntax_t1)
        count += 1

    # Ponieważ stringi są identyczne, replace wyżej mógł załatwić oba,
    # ale dla pewności puszczamy replace jeszcze raz, jeśli zostało.
    if bad_syntax_it2 in content:
        content = content.replace(bad_syntax_it2, good_syntax_it2)
        count += 1

    print(f"[INFO] Naprawiono {count} wystąpień błędnego docstringa.")

    # --- ZAPIS ---
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("-" * 50)
    print("[SUKCES] Biblioteka została naprawiona w pliku:")
    print(target_file)
    print("Uruchom teraz swój główny projekt.")
    print("-" * 50)


if __name__ == "__main__":
    patch_library_absolute()