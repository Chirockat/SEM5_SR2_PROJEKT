import ast
import numpy as np
import radon.raw as raw
from radon.complexity import cc_visit
import textwrap
from typing import Tuple
from pyit2fls import IT2FS, TSK, einstein_sum_s_norm, product_t_norm, trapezoid_mf, IT2TSK


def evaluate_code(filename: str) -> Tuple[float, float]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        return 0.0, 0.0
    except OSError:
        clean_name = filename.strip('{}')
        try:
            with open(clean_name, 'r', encoding='utf-8') as f:
                code = f.read()
        except:
            return 0.0, 0.0

    # 1. Ścieżka A: Liczymy CC dla istniejących funkcji
    defined_cc = 0.0
    blocks = []
    try:
        blocks = cc_visit(code)
        if blocks:
            defined_cc = sum(block.complexity for block in blocks) / len(blocks)
    except Exception:
        pass

    # 2. Ścieżka B: Liczymy CC dla kodu globalnego
    script_cc = 0.0
    try:
        wrapped_code = "def _global_wrapper():\n" + textwrap.indent(code, "    ")
        wrapped_blocks = cc_visit(wrapped_code)
        if wrapped_blocks:
            script_cc = wrapped_blocks[0].complexity
    except Exception:
        pass

    # 3. Wybieramy wynik

    final_cc = max(defined_cc, script_cc)

    if final_cc == 0:
        final_cc = 1.0

    # 4. Liczymy LLOC
    try:
        raw_metrics = raw.analyze(code)
        lines_count = raw_metrics.lloc
    except Exception:
        lines_count = len([line for line in code.splitlines() if line.strip()])

    if lines_count == 0:
        return 0.0, final_cc

    # 5. Parsujemy AST i liczymy Density
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0, final_cc

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1

    ast_density = node_count / lines_count

    return ast_density, final_cc


# --- 2. KLASA STEROWNIKA ROZMYTEGO ---
class FuzzyQualityController:
    def __init__(self):
        self._setup_system()

    def _setup_system(self):
        # Definicja uniwersów
        density_universe = np.linspace(0, 40, 2000)
        cc_universe = np.linspace(0, 50, 5000)

        d_low = IT2FS(density_universe, trapezoid_mf, [-0.1, 0, 4, 8, 1.0], trapezoid_mf, [-0.1, 0, 2, 5, 1.0])
        d_optimal = IT2FS(density_universe, trapezoid_mf, [3, 6, 10, 14, 1.0], trapezoid_mf, [5, 7, 8, 11, 1.0])
        d_high = IT2FS(density_universe, trapezoid_mf, [10, 25, 40, 40.1, 1.0], trapezoid_mf, [13, 30, 40, 40.1, 1.0])

        c_low = IT2FS(cc_universe, trapezoid_mf, [-0.1, 0, 5, 10, 1.0], trapezoid_mf, [-0.1, 0, 3, 6, 1.0])
        c_medium = IT2FS(cc_universe, trapezoid_mf, [5, 10, 15, 25, 1.0], trapezoid_mf, [8, 12, 14, 18, 1.0])
        c_high = IT2FS(cc_universe, trapezoid_mf, [15, 25, 50, 50.1, 1.0], trapezoid_mf, [22, 30, 50, 50.1, 1.0])

        self.controller = IT2TSK(t_norm=product_t_norm, s_norm=einstein_sum_s_norm)

        self.controller.add_input_variable('density')
        self.controller.add_input_variable('complexity')
        self.controller.add_output_variable('quality')

        # Reguły: Zmodyfikowane na strukturę przedziałową (dwa słowniki) oraz z wartościami z poprawnego skryptu

        # --- GRUPA 1 ---
        self.controller.add_rule(
            [('density', d_low), ('complexity', c_low)],
            [('quality', {'const': 78, 'density': 0.0, 'complexity': -2.0},
              {'const': 88, 'density': 0.0, 'complexity': -2.0})]
        )
        self.controller.add_rule(
            [('density', d_optimal), ('complexity', c_low)],
            [('quality', {'const': 100, 'density': 0.0, 'complexity': -4.7},
              {'const': 100, 'density': 0.0, 'complexity': -4.7})]
        )
        self.controller.add_rule(
            [('density', d_high), ('complexity', c_low)],
            [('quality', {'const': 107, 'density': -1.5, 'complexity': -2.0},
              {'const': 117, 'density': -1.5, 'complexity': -2.0})]
        )

        # --- GRUPA 2 ---
        self.controller.add_rule(
            [('density', d_low), ('complexity', c_medium)],
            [('quality', {'const': 74, 'density': 0.0, 'complexity': -2.0},
              {'const': 84, 'density': 0.0, 'complexity': -2.0})]
        )
        self.controller.add_rule(
            [('density', d_optimal), ('complexity', c_medium)],
            [('quality', {'const': 84, 'density': 0.0, 'complexity': -2.0},
              {'const': 94, 'density': 0.0, 'complexity': -2.0})]
        )
        self.controller.add_rule(
            [('density', d_high), ('complexity', c_medium)],
            [('quality', {'const': 42.5, 'density': -0.5, 'complexity': -1.0},
              {'const': 52.5, 'density': -0.5, 'complexity': -1.0})]
        )

        # --- GRUPA 3 ---
        self.controller.add_rule(
            [('density', d_low), ('complexity', c_high)],
            [('quality', {'const': 50, 'density': 0.0, 'complexity': -1.0},
              {'const': 60, 'density': 0.0, 'complexity': -1.0})]
        )
        self.controller.add_rule(
            [('density', d_optimal), ('complexity', c_high)],
            [('quality', {'const': 50, 'density': 0.0, 'complexity': -1.0},
              {'const': 60, 'density': 0.0, 'complexity': -1.0})]
        )
        self.controller.add_rule(
            [('density', d_high), ('complexity', c_high)],
            [('quality', {'const': 45, 'density': -0.5, 'complexity': -0.5},
              {'const': 45, 'density': -0.5, 'complexity': -0.5})]
        )

    def calculate_score(self, density: float, complexity: float) -> float:
        try:
            result = self.controller.evaluate({'density': density, 'complexity': complexity})

            score = result['quality']

            return max(0, min(100, float(score)))
        except Exception as e:
            print(f"Błąd obliczeń IT2TSK: {e}")
            return 0.0
quality_system = FuzzyQualityController()