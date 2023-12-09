import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SEARCH_X_LIMS = [-1000, 1000]
SEARCH_Y_LIMS = [-1000, 1000]
MAX_SEARCH_POINTS = 500
MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION = 50
MAX_ITERS = 1000
EPSILON = 0.0001
X_PIXELS = int(3440/40)
Y_PIXELS = int(1440/40)


def points_approx_equal(p1, p2, EPSILON):
    return np.linalg.norm([p1[x]-p2[x], p1[y]-p2[y]]) < 2*EPSILON

def get_F_sym(f, g):
    return [f, g]

def get_J_sym(expressions, symbols):
    return [[sp.diff(exp, sym) for sym in symbols] for exp in expressions]

def get_J_num(j_sym, substitutions):
    all_free_symbols = set()
    for row in j_sym:
        for expr in row:
            for sym in expr.free_symbols:
                all_free_symbols.add(sym)

    all_substitutions = [(symbol, substitutions[symbol])
                         for symbol in all_free_symbols
                         if symbol in substitutions.keys()]
    arr = sp.Array(j_sym).subs(all_substitutions)
    if len(arr.free_symbols) != 0:
        raise Exception(f'Did not provide substitutions for these free symbols: {arr.free_symbols}')
    return np.array(arr).astype(np.float64)

def get_F_num(f_sym, substitutions):
    arr = np.zeros([2, 1])
    for i in [0, 1]:
        all_substitutions = []
        for symbol in f_sym[i].free_symbols:
            if symbol in substitutions:
                all_substitutions.append((symbol, substitutions[symbol]))
            else:
                raise Exception(f'Did not provide substitutions for symbol: {symbol}')
        arr[i, 0] = f_sym[i].subs(all_substitutions)

    return arr

def newton_solve(f_sym, j_sym, starting_guess, ordered_syms, epsilon, max_iters):
    current_guess = starting_guess
    delta_norm_history = []
    delta_norm = 1000
    n_iters = 0
    while delta_norm > epsilon and n_iters < max_iters:
        current_guess, delta_norm = newton_iteration(f_sym, j_sym, current_guess, ordered_syms)
        delta_norm_history.append(delta_norm)
        n_iters += 1

    return current_guess, delta_norm_history

def newton_iteration(f_sym, j_sym, guess, ordered_syms):
    f_num = get_F_num(f_sym, guess)
    j_num = get_J_num(j_sym, guess)
    delta = np.linalg.solve(j_num, -f_num).flatten()
    new_guess = {sym: guess[sym] + delta[i] for i, sym in enumerate(ordered_syms)}
    delta_norm = np.linalg.norm(delta)
    return new_guess, delta_norm


x, y = sp.symbols('x y')
f_sym = get_F_sym(y**2+x**2-9, y-sp.sin(x)-2)
j_sym = get_J_sym(f_sym, [x, y])
# soln, delta_norm_history = newton_solve(f_sym, j_sym, {x: -5, y: 1}, [x, y], 0.00001, 1000)

# Do a randomised search to find unique solutions, stopping early if new unique solutions stop being found


converged_search_points_since_last_new_soln = 0
unique_solns = []
x_randoms = SEARCH_X_LIMS[0] + (SEARCH_X_LIMS[1]-SEARCH_X_LIMS[0])*np.random.random(MAX_SEARCH_POINTS)
y_randoms = SEARCH_Y_LIMS[0] + (SEARCH_Y_LIMS[1]-SEARCH_Y_LIMS[0])*np.random.random(MAX_SEARCH_POINTS)
point_count = 0
for x_rand, y_rand in zip(x_randoms, y_randoms):
    point_count += 1
    print(f'Searching point number {point_count}')
    soln, delta_norm_history = newton_solve(f_sym, j_sym, {x: x_rand, y: y_rand}, [x, y], EPSILON, MAX_ITERS)
    converged = len(delta_norm_history) < MAX_ITERS
    if converged:
        if not unique_solns:
            unique_solns.append(soln)
        if any([points_approx_equal(existing_soln, soln, EPSILON) for existing_soln in unique_solns]):
            converged_search_points_since_last_new_soln += 1
        else:
            converged_search_points_since_last_new_soln = 0
            unique_solns.append(soln)
        if converged_search_points_since_last_new_soln >= MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION:
            print(f'End search with {len(unique_solns)} unique solutions after reaching the limit of {MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION} consecutive converged search points since the last new unique solution')
            break

# Collapse to a grid of final image's aspect ratio and with some borders around the unique solutions
if len(unique_solns) < 2:
    print('Found fewer than 2 solutions, cannot generate an image')
soln_xs, soln_ys = [soln[x] for soln in unique_solns], [soln[y] for soln in unique_solns]
x_min, x_mean, x_max = np.min(soln_xs), np.mean(soln_xs), np.max(soln_xs)
y_min, y_mean, y_max = np.min(soln_ys), np.mean(soln_ys), np.max(soln_ys)
# Need to handle cases where solutions are collinear in x or y directions
if y_min == y_max:
    x_range = (x_max - x_min) * 4
    y_range = x_range * Y_PIXELS / X_PIXELS
elif x_min == x_max:
    y_range = (y_max - y_min) * 4
    x_range = y_range * X_PIXELS / Y_PIXELS
else:
    x_range = (x_max - x_min) * 4
    y_range = (y_max - y_min) * 4

x_coords = np.linspace(x_mean-x_range/2, x_mean+x_range/2, X_PIXELS)
y_coords = np.linspace(y_mean-y_range/2, y_mean+y_range/2, Y_PIXELS)

# Find the converged solution for each pixel - record it and the number of iterations
# Determine the base colours for each unique solution, and how they will be modified by number of iterations
soln_grid = []
for row_num, y_init in enumerate(y_coords):
    row = []
    print(f'Now computing row {row_num+1} of {Y_PIXELS}')
    for x_init in x_coords:
        soln, delta_norm_history = newton_solve(f_sym, j_sym, {x: x_init, y: y_init}, [x, y], EPSILON, MAX_ITERS)
        converged = len(delta_norm_history) < MAX_ITERS
        row.append((soln if converged else None, len(delta_norm_history)))
    soln_grid.append(row)


# Match each pixel to a known unique solution
# If it's a new solution, log warning
image_grid = []
for solution_row in soln_grid:
    row = []
    for soln, iters in solution_row:
        if not soln:
            row.append((None, iters))
        else:
            match = None
            for i, unique_soln in enumerate(unique_solns):
                if points_approx_equal(unique_soln, soln, EPSILON):
                    match = (i, iters)
                    row.append(match)
                    break
            if match:
                row.append(match)
            else:
                print(f'WARNING: Image will ignore a novel solution found on the grid: {soln}')
                row.append((None, iters))
    image_grid.append(row)

pixel_grid = np.zeros([Y_PIXELS, X_PIXELS, 4], dtype=int)
for j in range(Y_PIXELS):
    for i in range(X_PIXELS):
        if image_grid[j][i][0] == 0:
            pixel_grid[j, i, :] = [255, 0, 0, max(100, 255-image_grid[j][i][1]*10)]
        else:
            pixel_grid[j, i, :] = [0, 255, 0, max(100, 255-image_grid[j][i][1]*10)]
plt.figure()
plt.imshow(pixel_grid)
plt.show()

