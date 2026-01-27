import numpy as np
from pyit2fls import T1TSK, T1FS, tri_mf, trapezoid_mf, gaussian_mf, IT2FS, TSK, einstein_sum_s_norm, product_t_norm, IT2FS_plot
import matplotlib.pyplot as plt




density_universe = np.linspace(0, 40, 2000)
cc_universe = np.linspace(0, 50, 5000)


d_low = IT2FS(density_universe,
              trapezoid_mf, [-0.1, 0, 4, 8, 1.0],      # UMF:
              trapezoid_mf, [-0.1, 0, 2, 5, 1.0])      # LMF:


d_optimal = IT2FS(density_universe,
                  trapezoid_mf, [3, 6, 10, 14, 1.0],   # UMF:
                  trapezoid_mf, [5, 7, 8, 11, 1.0])    # LMF:


d_high = IT2FS(density_universe,
               trapezoid_mf, [10, 25, 40, 40.1, 1.0],  # UMF
               trapezoid_mf, [13, 30, 40, 40.1, 1.0])  # LMF



c_low = IT2FS(cc_universe,
              trapezoid_mf, [-0.1, 0, 5, 10, 1.0],     # UMF
              trapezoid_mf, [-0.1, 0, 3, 6, 1.0])      # LMF


c_medium = IT2FS(cc_universe,
                 trapezoid_mf, [5, 10, 15, 25, 1.0],   # UMF:
                 trapezoid_mf, [8, 12, 14, 18, 1.0])   # LMF:


c_high = IT2FS(cc_universe,
               trapezoid_mf, [15, 25, 50, 50.1, 1.0],  # UMF
               trapezoid_mf, [22, 30, 50, 50.1, 1.0])  # LMF

d_low.plot()
d_optimal.plot()
d_high.plot()

c_low.plot()
c_medium.plot()
c_high.plot()



def low_density_low_complexity(inputs):
    return 70 + 3 * inputs['density'] - 2 * inputs['complexity']

def optimal_density_low_complexity(inputs):
    return 100 - 2 * inputs['complexity']

def high_density_low_complexity(inputs):
    return 95 - 2 * inputs['density'] - 2 * inputs['complexity']

def low_density_medium_complexity(inputs):
    return 60 + 2 * inputs['density'] - 3 * inputs['complexity']

def optimal_density_medium_complexity(inputs):
    return 85 - 3 * inputs['complexity']

def high_density_medium_complexity(inputs):
    return 90 - 2.5 * inputs['density'] - 3 * inputs['complexity']

def low_density_high_complexity(inputs):
    return 45 + 1 * inputs['density'] - 1 * inputs['complexity']

def optimal_density_high_complexity(inputs):
    return 50 - 1 * inputs['complexity']

def high_density_high_complexity(inputs):
    return 60 - 1.5 * inputs['density'] - 1.5 * inputs['complexity']
# --- 4. STEROWNIK I REGUŁY ---

controller = TSK(t_norm=product_t_norm, s_norm=einstein_sum_s_norm)

controller.add_input_variable('density')
controller.add_input_variable('complexity')
controller.add_output_variable('quality')
# --- REGUŁY WNIOSKOWANIA (9 przypadków) ---

# --- GRUPA 1: NISKA ZŁOŻONOŚĆ (Dobry kod logicznie) ---

controller.add_rule(
    [('density', d_low), ('complexity', c_low)],
    [('quality', {'const': 70, 'density': 3, 'complexity': -2})]
)


controller.add_rule(
    [('density', d_optimal), ('complexity', c_low)],
    [('quality', {'const': 100, 'density': 0, 'complexity': -2})]
)


controller.add_rule(
    [('density', d_high), ('complexity', c_low)],
    [('quality', {'const': 95, 'density': -2, 'complexity': -2})]
)


# --- GRUPA 2: ŚREDNIA ZŁOŻONOŚĆ (Kod przeciętny logicznie) ---

controller.add_rule(
    [('density', d_low), ('complexity', c_medium)],
    [('quality', {'const': 60, 'density': 2, 'complexity': -3})]
)

controller.add_rule(
    [('density', d_optimal), ('complexity', c_medium)],
    [('quality', {'const': 85, 'density': 0, 'complexity': -3})]
)

controller.add_rule(
    [('density', d_high), ('complexity', c_medium)],
    [('quality', {'const': 90, 'density': -2.5, 'complexity': -3})]
)


# --- GRUPA 3: WYSOKA ZŁOŻONOŚĆ (Zły kod logicznie) ---


controller.add_rule(
    [('density', d_low), ('complexity', c_high)],
    [('quality', {'const': 45, 'density': 1, 'complexity': -1})]
)

controller.add_rule(
    [('density', d_optimal), ('complexity', c_high)],
    [('quality', {'const': 50, 'density': 0, 'complexity': -1})]
)


controller.add_rule(
    [('density', d_high), ('complexity', c_high)],
    [('quality', {'const': 60, 'density': -1.5, 'complexity': -1.5})]
)

# --- 5. SYMULACJA NA PRZYKŁADACH ---

print("-" * 50)
print(f"{'TYP KODU':<15} | {'Dens.':<5} | {'CC':<5} | {'OCENA (0-100)':<15}")
print("-" * 50)



# Scenariusz 1: "Knapsack Basic" (Ten zrównoważony)
# Density ok. 5-6, CC małe (np. 3)
case_basic = {'density': 6, 'complexity': 6}
res_basic = controller.evaluate(case_basic)
print(f"{'BASIC (Ideal)':<15} | {case_basic['density']:<5} | {case_basic['complexity']:<5} | {res_basic['quality']:.2f}")

# Scenariusz 2: "Knapsack Modern" (Zbyt gęsty)
# Density bardzo duże (np. 14), CC bardzo małe (np. 2)
case_modern = {'density': 14.0, 'complexity': 7}
res_modern = controller.evaluate(case_modern)
print(f"{'MODERN (Dense)':<15} | {case_modern['density']:<5} | {case_modern['complexity']:<5} | {res_modern['quality']:.2f}")

# Scenariusz 3: "Knapsack Messy" (Rozwlekły + Ifologia)

case_messy = {'density': 6, 'complexity': 12}
res_messy = controller.evaluate(case_messy)
print(f"{'MESSY (Spag.)':<15} | {case_messy['density']:<5} | {case_messy['complexity']:<5} | {res_messy['quality']:.2f}")

# Scenariusz 4: "God Class" (Najgorszy przypadek)
# Wszystko źle
case_bad = {'density': 15.0, 'complexity': 15}
res_bad = controller.evaluate(case_bad)
print(f"{'BAD CODE':<15} | {case_bad['density']:<5} | {case_bad['complexity']:<5} | {res_bad['quality']:.2f}")

print("-" * 50)

# ---  WYKRES POWIERZCHNI STEROWANIA (3D) ---

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0, 20, 50)  # Density
y = np.linspace(0, 20, 50)  # Complexity
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(x)):
    for j in range(len(y)):
        res = controller.evaluate({'density': X[i, j], 'complexity': Y[i, j]})
        Z[i, j] = res['quality']

surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_xlabel('AST Density (Gęstość)')
ax.set_ylabel('Complexity (CC)')
ax.set_zlabel('Ocena Jakości')
ax.set_title('Ocena kodu: Zależność od Gęstości i CC')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
