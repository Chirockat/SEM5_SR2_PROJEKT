import ast
import radon.raw as raw
import radon.complexity as cc


def analyze_ast_density(filename):
    print(f"--- Analiza pliku: {filename} ---")

    with open(filename, 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. Obliczanie Złożoności (CC)
    blocks = cc.cc_visit(code)

    print("\n[1. Złożoność Cyklomatyczna - CC]")
    total_cc = 0
    if blocks:
        for block in blocks:
            print(f"  Funkcja '{block.name}' -> CC: {block.complexity}")
            total_cc += block.complexity
        avg_cc = total_cc / len(blocks)
    else:
        avg_cc = 0
    print(f"-> ŚREDNIE CC: {avg_cc:.2f}")

    # 1. Liczymy linie kodu (korzystamy z Radona, bo dobrze ignoruje komentarze)
    try:
        raw_metrics = raw.analyze(code)
        # Używamy SLOC (Source Lines of Code) lub LLOC.
        lines_count = raw_metrics.lloc
    except:
        lines_count = len(code.splitlines())

    if lines_count == 0:
        return

    # 2. Parsujemy kod do drzewa AST
    try:
        tree = ast.parse(code)
    except SyntaxError:
        print("Błąd składni w pliku!")
        return

    # 3. Liczymy WSZYSTKIE węzły syntaktyczne
    # To policzy każdą zmienną, każde dodawanie, każde wywołanie funkcji
    node_count = 0
    for node in ast.walk(tree):
        node_count += 1

    # 4. Wyliczamy Gęstość AST
    ast_density = node_count / lines_count

    print(f"Liczba linii (LLOC): {lines_count}")
    print(f"Liczba węzłów AST (Node Count): {node_count}")
    print(f"-> AST DENSITY (Węzły na linię): {ast_density:.2f}")

# Sprawdźmy wszystkie 3 pliki
files = ['knapsack_modern.py', 'knapsack_basic.py', 'knapsack_messy.py']

for f in files:
    try:
        analyze_ast_density(f)
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {f}")