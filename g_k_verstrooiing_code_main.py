# Name: Monsieurs Pieter
# E-mail: @s0224946@ad.ua.ac.be
# Title: Main code voor gevorderde kwantum taak - 3D Harmonische Oscillator Matching Method
# Last edit: 8/11/2026
# Status: bezig, taak 1

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

print("hello world")

######################################## Task 1 ###########################################

# =============================================================================
# DE DRIE NUMEROV BROERS
# =============================================================================

# Klassieke Numerov
####################

def classical_numerov_forward(x, k2, y0, y1):
    n = len(x)
    h = x[1] - x[0]
    h2_12 = (h**2) / 12.0
    y = np.zeros(n)
    y[0], y[1] = y0, y1
    for i in range(1, n - 1):
        # Big fraction
        factor_curr = 2.0 * (1.0 - 5.0 * h2_12 * k2[i]) * y[i]
        factor_prev = (1.0 + h2_12 * k2[i-1]) * y[i-1]
        factor_next = 1.0 + h2_12 * k2[i+1]
        y[i+1] = (factor_curr - factor_prev) / factor_next
    return y

def classical_numerov_backward(x, k2, y_end, y_next):
    n = len(x)
    h = x[1] - x[0]
    h2_12 = (h**2) / 12.0
    y = np.zeros(n)
    y[-1], y[-2] = y_end, y_next
    for i in range(n - 2, 0, -1):
        factor_curr = 2.0 * (1.0 - 5.0 * h2_12 * k2[i]) * y[i]
        factor_next = (1.0 + h2_12 * k2[i+1]) * y[i+1]
        factor_prev = 1.0 + h2_12 * k2[i-1]
        y[i-1] = (factor_curr - factor_next) / factor_prev
    return y

# Matrix Numerov
#################
# Eerste punt moet gelijk zijn aan uw stapgroote, dna zou het moeen lukken
def matrix_numerov(x, V_mat, is_eigenvalue_problem=True):
    n = len(x)
    h = x[1] - x[0]
    
    # Controleer of het eerste roosterpunt inderdaad gelijk is aan de stapgrootte h
    if not np.isclose(x[0], h):
        raise ValueError(f"Eerste punt x[0] ({x[0]}) is niet gelijk aan stapgrootte h ({h})!")
        
    # Omdat x[0] = h, is x[0] al het eerste INTERNE punt (r = 0 ligt één stap h voor x[0]).
    # Alleen het laatste punt x[-1] (r_max) heeft een randvoorwaarde u(r_max) = 0, dus n - 1 interne punten:
    num_internal = n - 1
    
    # Prepare A
    A = np.zeros((num_internal, num_internal))
    np.fill_diagonal(A, -2.0)
    np.fill_diagonal(A[1:], 1.0)
    np.fill_diagonal(A[:, 1:], 1.0)
    A = A / (h**2)
    
    # Prepare B
    B = np.zeros((num_internal, num_internal))
    np.fill_diagonal(B, 10.0 / 12.0)
    np.fill_diagonal(B[1:], 1.0 / 12.0)
    np.fill_diagonal(B[:, 1:], 1.0 / 12.0)
    
    # Bouw de Hamiltoniaan op (gebruik V_mat[:-1] omdat x[0] t/m x[-2] interne punten zijn)
    H = -0.5 * (la.inv(B) @ A) + np.diag(V_mat[:-1])
    H = 0.5 * (H + H.T)
    
    # Los de Schrödingervergelijking op
    energies, wavefunctions_internal = la.eigh(H)
    
    # Zet de golffuncties terug op het rooster (op x[-1] geldt u(r_max) = 0)
    wavefunctions = np.zeros((n, num_internal))
    wavefunctions[:-1, :] = wavefunctions_internal
    
    return energies, wavefunctions


# Generalized Numerov
######################

def generalized_numerov_prof_forward(r, F_array, u0, u1):
    """STRIKT VOLGENS VERGELIJKING 14, 15 EN 16 VAN DE PROF (FORWARD)"""
    n = len(r)
    h = r[1] - r[0]
    u = np.zeros(n)
    w = np.zeros(n)
    
    u[0], u[1] = u0, u1
    
    # Equation 14, auxilary variable w_i
    # Forward way
    w[0] = (1.0 - (h**2 / 12.0) * F_array[0]) * u[0]
    w[1] = (1.0 - (h**2 / 12.0) * F_array[1]) * u[1]
    
    for i in range(1, n - 1):
        # Equation 15, updating w
        # Forward way
        w[i+1] = 2.0 * w[i] - w[i-1] + (h**2) * F_array[i] * u[i]
        # Equation 16, we revert back to u_i
        # Forward way
        u[i+1] = w[i+1] / (1.0 - (h**2 / 12.0) * F_array[i+1])
    return u

def generalized_numerov_prof_backward(r, F_array, u_end, u_next):
    """STRIKT VOLGENS VERGELIJKING 14, 15 EN 16 VAN DE PROF (BACKWARD)"""
    n = len(r)
    h = r[1] - r[0]
    u = np.zeros(n)
    w = np.zeros(n)
    
    u[-1], u[-2] = u_end, u_next
    
    # Equation 14, auxilary variable w_i
    # Backward way
    w[-1] = (1.0 - (h**2 / 12.0) * F_array[-1]) * u[-1]
    w[-2] = (1.0 - (h**2 / 12.0) * F_array[-2]) * u[-2]
    
    for i in range(n - 2, 0, -1):
        # Equation 15, updating w
        # Backward way
        w[i-1] = 2.0 * w[i] - w[i+1] + (h**2) * F_array[i] * u[i]
        # Equation 16, we revert back to u_i
        # Backward way
        u[i-1] = w[i-1] / (1.0 - (h**2 / 12.0) * F_array[i-1])
    return u

# =============================================================================
# MATCHING METHOD LOGICA
# =============================================================================

# Time to glue the "forward" and "backward" method together.
# Classical and Generalised need it, while Matrix doesn't.

def matching_error(E, r_grid, V_eff):
    E_val = E[0]
    k2 = 2.0 * (E_val - V_eff)
    h = r_grid[1] - r_grid[0]
    match_idx = np.searchsorted(r_grid, 1.73)
    
    r_forward = r_grid[:match_idx+2]
    u_forward = classical_numerov_forward(r_forward, k2[:match_idx+2], 0.0, 1e-5)
    
    r_backward = r_grid[match_idx-1:]
    u_backward = classical_numerov_backward(r_backward, k2[match_idx-1:], 0.0, 1e-10)
    
    scale = u_forward[match_idx] / u_backward[1]
    u_backward *= scale
    
    deriv_forward = (u_forward[match_idx+1] - u_forward[match_idx-1]) / (2.0 * h)
    deriv_backward = (u_backward[2] - u_backward[0]) / (2.0 * h)
    return deriv_forward - deriv_backward

def prof_matching_error(E, r_grid, V_eff):
    E_val = E[0]
    F_array = 2.0 * (V_eff - E_val)
    h = r_grid[1] - r_grid[0]
    match_idx = np.searchsorted(r_grid, 1.73)
    
    r_forward = r_grid[:match_idx+2]
    u_forward = generalized_numerov_prof_forward(r_forward, F_array[:match_idx+2], 0.0, 1e-5)
    
    r_backward = r_grid[match_idx-1:]
    u_backward = generalized_numerov_prof_backward(r_backward, F_array[match_idx-1:], 0.0, 1e-10)
    
    scale = u_forward[match_idx] / u_backward[1]
    u_backward *= scale
    
    deriv_forward = (u_forward[match_idx+1] - u_forward[match_idx-1]) / (2.0 * h)
    deriv_backward = (u_backward[2] - u_backward[0]) / (2.0 * h)
    return deriv_forward - deriv_backward

# =============================================================================
# HOOFDPROGRAMMA
# =============================================================================

# Time to run the code we built

if __name__ == "__main__":
    n_points = 700
    n_points = 700
    r_max = 5.5
    h = r_max / n_points
  
    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 4.0, n_points)
    # 4 as ameximum because of you clculate it it, it reaches four maximally
    # kwantummechanische harmonische oscillator
    # (een deeltje in een parabolische potentiaalput)
    # The potential for hbar = m = omega = 1, l = 0
    V_eff = 0.5 * r_grid**2
    match_idx = np.searchsorted(r_grid, 1.73)
    
    print(f"\n--- BEREKENING ENERGIENIVEAUS ---")
    
    # Matrix-Numerov
    ################
    # Schrodinger oplossen
    energies_matrix, wf_matrix = matrix_numerov(r_grid, V_eff)
    # The first energy, so that's the ground state energy
    E_matrix = energies_matrix[0]
    # Print on screen. Theoretically, we expect E = hbar omega*3/2 = 3/2 = 1.5
    # 1/2 hbar omega + 1/2 hbar omega + 1/2  hbar omega = 3/2 hbar omega = 3/2 = 1.5
    print(f"1. Matrix-Numerov E0: {E_matrix:.6f} (Theorie: 1.5) | Fout: {abs(E_matrix - 1.5):.4e}")
    
    # Klassieke Numerov
    ###################
    # We give him a good guess.
    E_guess = [1.4]
    # Schrodinger oplossen, with the glued together methods.
    E_classical_arr = fsolve(matching_error, E_guess, args=(r_grid, V_eff))
    E_classical = E_classical_arr[0]
    print(f"2. Shooting Numerov  E0: {E_classical:.6f} (Theorie: 1.5) | Fout: {abs(E_classical - 1.5):.4e}")
    
    # Prof's Gegeneraliseerde Numerov
    #################################
    # Solve Schrodinger, with glued together methods.
    E_prof_arr = fsolve(prof_matching_error, E_guess, args=(r_grid, V_eff))
    E_prof = E_prof_arr[0]
    print(f"3. Prof's Numerov    E0: {E_prof:.6f} (Theorie: 1.5) | Fout: {abs(E_prof - 1.5):.4e}")
    
    # --- ANALYTISCHE GOLFFUNCTIE OPLOSSING ---
    u_analytical = r_grid * np.exp(-r_grid**2 / 2.0)
    u_analytical /= np.max(np.abs(u_analytical))
    
    # --- 1. MATRIX GOLFFUNCTIE (+ SPIEGEL FIX) ---
    u_matrix = wf_matrix[:, 0]
    u_matrix /= np.max(np.abs(u_matrix))
    # Solution could be mirrored around the x-axis, so flip it back.
    if np.sum(u_matrix * u_analytical) < 0: 
        u_matrix = -u_matrix
    
    # --- 2. KLASSIEKE Numerov GOLFFUNCTIE ---
    k2_opt = 2.0 * (E_classical - V_eff)
    u_f_cl = classical_numerov_forward(r_grid[:match_idx+1], k2_opt[:match_idx+1], 0.0, 1e-5)
    u_b_cl = classical_numerov_backward(r_grid[match_idx-1:], k2_opt[match_idx-1:], 0.0, 1e-10)
    u_b_cl *= (u_f_cl[-1] / u_b_cl[1])
    
    u_shoot_classical = np.concatenate([u_f_cl[:-1], u_b_cl[1:]])
    u_shoot_classical /= np.max(np.abs(u_shoot_classical))
    if np.sum(u_shoot_classical * u_analytical) < 0: 
        u_shoot_classical = -u_shoot_classical

    # --- 3. PROF'S GEGENERALISEERDE numerove gOLFFUNCTIE ---
    F_opt = 2.0 * (V_eff - E_prof)
    u_f_prof = generalized_numerov_prof_forward(r_grid[:match_idx+1], F_opt[:match_idx+1], 0.0, 1e-5)
    u_b_prof = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_opt[match_idx-1:], 0.0, 1e-10)
    u_b_prof *= (u_f_prof[-1] / u_b_prof[1])
    
    u_shoot_prof = np.concatenate([u_f_prof[:-1], u_b_prof[1:]])
    u_shoot_prof /= np.max(np.abs(u_shoot_prof))
    if np.sum(u_shoot_prof * u_analytical) < 0: 
        u_shoot_prof = -u_shoot_prof
    
    # --- VISUALISATIE MET PLOT---
    plt.figure(figsize=(9, 5.5))
    plt.plot(r_grid, u_analytical, 'k-', linewidth=6, label="Exact (Analytisch)", alpha=0.3)
    plt.plot(r_grid, u_matrix, 'b-', linewidth=2.5, label="1. Matrix-Numerov")
    plt.plot(r_grid, u_shoot_classical, 'g:', linewidth=2.5, label="2. Klassiek Numerov")
    plt.plot(r_grid, u_shoot_prof, 'r--', linewidth=2, label="3. Prof's Numerov (Gegeneraliseerd)")
    
    plt.title("Radiale Golffunctie $u(r)$ via de Matching Method (Ground state: $E_0=1.5$)")
    plt.xlabel("Radiale afstand $r$")
    plt.ylabel("$u(r)$")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    l = 1  # Stel impulsmoment in op l = 1
    n_points = 700
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 5.5, n_points)
    
    # Voeg de centrifugale barrière l(l+1) / (2r^2) toe aan V_eff
    V_eff = 0.5 * r_grid**2 + (l * (l + 1)) / (2.0 * r_grid**2)
    
    # Nieuwe guesses en theoretische waarden voor l = 1: E = 2*n_r + l + 1.5
    guesses_l1 = [2.4, 4.4, 6.4]
    theorie_l1 = [2.5, 4.5, 6.5]
    
    print(f"\n--- BEREKENING ENERGIENIVEAUS VOOR l = {l} ---")
    
    # 1. Matrix-Numerov
    energies_matrix, wf_matrix = matrix_numerov(r_grid, V_eff)
    
    # Gebruik r = 2.0 als matchpunt voor l = 1 (i.p.v. 1.73)
    match_idx = np.searchsorted(r_grid, 2.0)
    
    # Exacte analytische golffuncties voor l = 1 (u(r) ~ r^2 * Laguerre * exp(-r^2/2))
    u_exact_l1 = [
        (r_grid**2) * np.exp(-r_grid**2 / 2.0),
        (r_grid**2) * (5.0 - 2.0 * r_grid**2) * np.exp(-r_grid**2 / 2.0),
        (r_grid**2) * (35.0 - 28.0 * r_grid**2 + 4.0 * r_grid**4) * np.exp(-r_grid**2 / 2.0)
    ]
    for n in range(3):
        u_exact_l1[n] /= np.max(np.abs(u_exact_l1[n]))

    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']

    for n in range(3):
        E_mat = energies_matrix[n]
        
        # 2. Prof's Generalized Numerov met juiste randvoorwaarde u(r) ~ r^(l+1)
        def prof_error_l(E):
            E_val = E[0]
            F_arr = 2.0 * (V_eff - E_val)
            h = r_grid[1] - r_grid[0]
            u0, u1 = r_grid[0]**(l+1), r_grid[1]**(l+1)
            u_f = generalized_numerov_prof_forward(r_grid[:match_idx+2], F_arr[:match_idx+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_arr[match_idx-1:], 0.0, 1e-10)
            u_b *= (u_f[match_idx] / u_b[1])
            d_f = (u_f[match_idx+1] - u_f[match_idx-1]) / (2.0 * h)
            d_b = (u_b[2] - u_b[0]) / (2.0 * h)
            return d_f - d_b

        E_pr = fsolve(prof_error_l, [guesses_l1[n]])[0]
        
        print(f"\nToestand n_r = {n} (l = {l}):")
        print(f"  Matrix-Numerov: E = {E_mat:.6f} | Fout: {abs(E_mat - theorie_l1[n]):.2e}")
        print(f"  Prof Methode  : E = {E_pr:.6f} | Fout: {abs(E_pr - theorie_l1[n]):.2e}")
        
        # Golffunctie reconstrueren en plotten
        F_opt = 2.0 * (V_eff - E_pr)
        u_f = generalized_numerov_prof_forward(r_grid[:match_idx+1], F_opt[:match_idx+1], r_grid[0]**(l+1), r_grid[1]**(l+1))
        u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_opt[match_idx-1:], 0.0, 1e-10)
        u_b *= (u_f[-1] / u_b[1])
        u_shoot = np.concatenate([u_f[:-1], u_b[1:]])
        u_shoot /= np.max(np.abs(u_shoot))
        if np.sum(u_shoot * u_exact_l1[n]) < 0: u_shoot = -u_shoot
        
        plt.plot(r_grid, u_shoot, color=colors[n], linewidth=2, label=f"$n_r={n}, l=1$ ($E = {E_pr:.4f}$)")
        plt.plot(r_grid, u_exact_l1[n], 'k--', alpha=0.4, label="Analytisch" if n==0 else "")

    plt.title(f"3D Harmonische Oscillator ($l = {l}$) - Radiale Golffuncties", fontweight='bold')
    plt.xlabel("Radiale afstand $r$")
    plt.ylabel("$u(r)$")
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    n_points = 800
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 6.0, n_points)
    # h = r_grid[1] - r_grid[0]  # Correcte stapgrootte!
    
    # De 6 gecombineerde toestanden: E_0 t/m E_5 (afwisselend l=0 en l=1)
    # Format: (label, l, n_r, E_theorie, E_guess, r_match)
    states_info = [
        ("$E_0 = 1.5$\n($l=0, n_r=0$)", 0, 0, 1.5, 1.48, 1.73),
        ("$E_1 = 2.5$\n($l=1, n_r=0$)", 1, 0, 2.5, 2.48, 2.00),
        ("$E_2 = 3.5$\n($l=0, n_r=1$)", 0, 1, 3.5, 3.48, 1.73),
        ("$E_3 = 4.5$\n($l=1, n_r=1$)", 1, 1, 4.5, 4.48, 2.00),
        ("$E_4 = 5.5$\n($l=0, n_r=2$)", 0, 2, 5.5, 5.48, 1.73),
        ("$E_5 = 6.5$\n($l=1, n_r=2$)", 1, 2, 6.5, 6.48, 2.00),
    ]

    # Bereken Matrix-Numerov voor l=0 en l=1
    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)
    
    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)

    labels = []
    err_matrix = []
    err_prof = []

    print(f"{'Toestand':<16} | {'E_theorie':<9} | {'E_matrix':<10} | {'E_prof':<10} | {'|dE| Matrix':<11} | {'|dE| Prof':<11}")
    print("-" * 80)

    for label, l, n_r, E_th, guess, r_m in states_info:
        labels.append(label)
        V_eff = V_eff_0 if l == 0 else V_eff_1
        
        # 1. Matrix-Numerov waarde en absolute fout |E_num - E_theorie|
        E_m = E_mat_l0[n_r] if l == 0 else E_mat_l1[n_r]
        diff_m = abs(E_m - E_th)
        err_matrix.append(diff_m)
        
        # 2. Prof's Generalized Numerov shooting functie
        match_idx = np.searchsorted(r_grid, r_m)
        
        def prof_error_func(E_arr):
            E_val = E_arr[0]
            F_arr = 2.0 * (V_eff - E_val)
            
            # Randvoorwaarden bij r -> 0: u(r) ~ r^(l+1)
            u0 = r_grid[0]**(l + 1)
            u1 = r_grid[1]**(l + 1)
            
            u_f = generalized_numerov_prof_forward(r_grid[:match_idx+2], F_arr[:match_idx+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_arr[match_idx-1:], 0.0, 1e-10)
            
            # Schaal achterwaartse oplossing zodat u_b[1] gelijk is aan u_f[match_idx]
            u_b *= (u_f[match_idx] / u_b[1])
            
            # Centrale afgeleiden op het matchpunt
            d_f = (u_f[match_idx+1] - u_f[match_idx-1]) / (2.0 * h)
            d_b = (u_b[2] - u_b[0]) / (2.0 * h)
            return [d_f - d_b]

        E_p = fsolve(prof_error_func, [guess])[0]
        diff_p = abs(E_p - E_th)
        err_prof.append(diff_p)
        
        print(f"E_{int(E_th - 1.5)} (l={l}, n={n_r})  | {E_th:<9.1f} | {E_m:<10.6f} | {E_p:<10.6f} | {diff_m:<11.2e} | {diff_p:<11.2e}")

    # --- GROEPS-STAAFDIAGRAM (BAR CHART) ---
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width/2, err_matrix, width, label='Matrix Numerov', color='steelblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, err_prof, width, label="Prof's Generalized Numerov", color='darkorange', edgecolor='black')

    ax.set_yscale('log')
    ax.set_ylabel(r"Absolute fout $|E_{\mathrm{num}} - E_{\mathrm{theorie}}|$", fontsize=11)
    ax.set_title("Absolute Fout vs. Theorie: Matrix Numerov vs. Generalized Numerov ($E_0$ t/m $E_5$)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(1e-13, 1e-3)
    ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.35)
    ax.legend(fontsize=10.5)

    # Zet de exacte foutwaarde boven elke balk
    for rect in rects1:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}",
                    xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='navy')

    for rect in rects2:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}",
                    xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='darkred')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_points = 700
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 5.5, n_points)
    # h = r_grid[1] - r_grid[0]
    
    # Format: (label, l, n_r, E_theorie, E_guess, r_match)
    states_info = [
        ("$E_0 = 1.5$\n($l=0, n_r=0$)", 0, 0, 1.5, 1.48, 1.73),
        ("$E_1 = 2.5$\n($l=1, n_r=0$)", 1, 0, 2.5, 2.48, 2.00),
        ("$E_2 = 3.5$\n($l=0, n_r=1$)", 0, 1, 3.5, 3.48, 1.73),
        ("$E_3 = 4.5$\n($l=1, n_r=1$)", 1, 1, 4.5, 4.48, 2.00),
        ("$E_4 = 5.5$\n($l=0, n_r=2$)", 0, 2, 5.5, 5.48, 1.73),
        ("$E_5 = 6.5$\n($l=1, n_r=2$)", 1, 2, 6.5, 6.48, 2.00),
    ]

    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)
    
    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)

    labels = []
    err_matrix = []
    err_prof = []

    print(f"{'Toestand':<16} | {'E_theorie':<9} | {'E_matrix':<10} | {'E_prof':<10} | {'|dE| Matrix':<11} | {'|dE| Prof':<11}")
    print("-" * 80)

    for label, l, n_r, E_th, guess, r_m in states_info:
        labels.append(label)
        V_eff = V_eff_0 if l == 0 else V_eff_1
        
        # 1. Matrix-Numerov
        E_m = E_mat_l0[n_r] if l == 0 else E_mat_l1[n_r]
        diff_m = abs(E_m - E_th)
        err_matrix.append(diff_m)
        
        # 2. Prof's Generalized Numerov
        match_idx = np.searchsorted(r_grid, r_m)
        
        def prof_error_func(E_arr):
            E_val = E_arr[0]
            F_arr = 2.0 * (V_eff - E_val)
            
            u0 = r_grid[0]**(l + 1)
            u1 = r_grid[1]**(l + 1)
            
            u_f = generalized_numerov_prof_forward(r_grid[:match_idx+2], F_arr[:match_idx+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_arr[match_idx-1:], 0.0, 1e-10)
            
            # Schaal achterwaartse oplossing op het matchpunt (u_b[1] komt overeen met match_idx)
            u_b *= (u_f[match_idx] / u_b[1])
            
            # Centrale afgeleiden op het matchpunt
            d_f = (u_f[match_idx+1] - u_f[match_idx-1]) / (2.0 * h)
            d_b = (u_b[2] - u_b[0]) / (2.0 * h)
            return [d_f - d_b]

        E_p = fsolve(prof_error_func, [guess])[0]
        diff_p = abs(E_p - E_th)
        err_prof.append(diff_p)
        
        print(f"E_{int(E_th - 1.5)} (l={l}, n={n_r})  | {E_th:<9.1f} | {E_m:<10.6f} | {E_p:<10.6f} | {diff_m:<11.2e} | {diff_p:<11.2e}")

    # --- GROEPS-STAAFDIAGRAM (BAR CHART) ---
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width/2, err_matrix, width, label='Matrix Numerov', color='steelblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, err_prof, width, label="Prof's Generalized Numerov", color='darkorange', edgecolor='black')

    ax.set_yscale('log')
    ax.set_ylabel(r"Absolute fout $|E_{\mathrm{num}} - E_{\mathrm{theorie}}|$", fontsize=11)
    ax.set_title("Absolute Fout vs. Theorie: Matrix Numerov vs. Generalized Numerov ($E_0$ t/m $E_5$)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(1e-13, 1e-3)
    ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.35)
    ax.legend(fontsize=10.5)

    for rect in rects1:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}",
                    xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='navy')

    for rect in rects2:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}",
                    xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='darkred')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_points = 700
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 5.5, n_points)
    # h = np.take(r_grid, 1) - np.take(r_grid, 0)
    
    states_info = [
        ("E0 = 1.5\n(l=0, nr=0)", 0, 0, 1.5, 1.48, 1.73),
        ("E1 = 2.5\n(l=1, nr=0)", 1, 0, 2.5, 2.48, 2.00),
        ("E2 = 3.5\n(l=0, nr=1)", 0, 1, 3.5, 3.48, 1.73),
        ("E3 = 4.5\n(l=1, nr=1)", 1, 1, 4.5, 4.48, 2.00),
        ("E4 = 5.5\n(l=0, nr=2)", 0, 2, 5.5, 5.48, 1.73),
        ("E5 = 6.5\n(l=1, nr=2)", 1, 2, 6.5, 6.48, 2.00),
    ]

    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)
    
    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)

    labels, err_matrix, err_prof = [], [], []

    print(f"{'Toestand':<15} | {'|dE| Matrix':<12} | {'|dE| Prof Gen.':<14}")
    print("-" * 48)

    for label, l, n_r, E_th, guess, r_m in states_info:
        labels.append(label)
        V_eff = V_eff_0 if l == 0 else V_eff_1
        
        # 1. Matrix-Numerov fout
        E_m = np.take(E_mat_l0 if l == 0 else E_mat_l1, n_r)
        diff_m = abs(E_m - E_th)
        err_matrix.append(diff_m)
        
        # 2. Prof's Generalized Numerov fout (via bestaande functies)
        match_idx = int(np.searchsorted(r_grid, r_m))
        
        def prof_err(E_arr):
            E_val = float(np.take(E_arr, 0))
            F_arr = 2.0 * (V_eff - E_val)
            u0 = np.take(r_grid, 0)**(l + 1)
            u1 = np.take(r_grid, 1)**(l + 1)
            
            u_f = generalized_numerov_prof_forward(r_grid[:match_idx+2], F_arr[:match_idx+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_arr[match_idx-1:], 0.0, 1e-10)
            
            # Schaal achterwaarts op het matchpunt (index 1 in u_b is match_idx)
            u_b = u_b * (np.take(u_f, match_idx) / np.take(u_b, 1))
            
            d_f = (np.take(u_f, match_idx + 1) - np.take(u_f, match_idx - 1)) / (2.0 * h)
            d_b = (np.take(u_b, 2) - np.take(u_b, 0)) / (2.0 * h)
            return [d_f - d_b]

        E_p = float(np.take(fsolve(prof_err, [guess]), 0))
        diff_p = abs(E_p - E_th)
        err_prof.append(diff_p)
        
        print(f"E_{int(E_th - 1.5)} (l={l}, n={n_r}) | {diff_m:<12.2e} | {diff_p:<14.2e}")

    # --- GROEPS-STAAFDIAGRAM ---
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width/2, err_matrix, width, label='Matrix Numerov', color='steelblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, err_prof, width, label="Prof's Generalized Numerov", color='darkorange', edgecolor='black')

    ax.set_yscale('log')
    ax.set_ylabel("Absolute fout |E_num - E_theorie|")
    ax.set_title("Nauwkeurigheid: Matrix Numerov vs. Generalized Numerov (E0 t/m E5)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(1e-13, 1e-3)
    ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.35)
    ax.legend()

    for rect in rects1:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}", (rect.get_x() + rect.get_width()/2, val),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8, color='navy')
    for rect in rects2:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}", (rect.get_x() + rect.get_width()/2, val),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8, color='darkred')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_points = 700
    r_max = 6.0  # Iets ruimer zodat ook E_5 = 6.5 niet tegen de rand botst
    h = r_max / n_points
    r_grid = np.linspace(h, r_max, n_points)  # Eerste punt is exact gelijk aan h!

    # De 6 toestanden: E_0 t/m E_5 (afwisselend l=0 en l=1)
    states_info = [
        ("E0 = 1.5\n(l=0, nr=0)", 0, 0, 1.5, 1.48, 1.73),
        ("E1 = 2.5\n(l=1, nr=0)", 1, 0, 2.5, 2.48, 2.00),
        ("E2 = 3.5\n(l=0, nr=1)", 0, 1, 3.5, 3.48, 1.73),
        ("E3 = 4.5\n(l=1, nr=1)", 1, 1, 4.5, 4.48, 2.00),
        ("E4 = 5.5\n(l=0, nr=2)", 0, 2, 5.5, 5.48, 1.73),
        ("E5 = 6.5\n(l=1, nr=2)", 1, 2, 6.5, 6.48, 2.00),
    ]

    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)

    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)

    labels = []
    err_matrix_list = []
    err_prof_list = []
    error_gap_list = []      # |err_Matrix - err_Prof|
    direct_diff_list = []    # |E_Matrix - E_Prof|

    print(f"{'Toestand':<15} | {'Fout Matrix':<12} | {'Fout Prof':<12} | {'Gap |dE1-dE2|':<14} | {'|E_mat - E_prof|':<16}")
    print("-" * 80)

    for label, l, n_r, E_th, guess, r_m in states_info:
        labels.append(label)
        V_eff = V_eff_0 if l == 0 else V_eff_1

        # 1. Matrix-Numerov
        E_m = float(np.take(E_mat_l0 if l == 0 else E_mat_l1, n_r))
        err_m = abs(E_m - E_th)
        err_matrix_list.append(err_m)

        # 2. Prof's Generalized Numerov (startend op r = 0 met u(0) = 0, u(h) = h^(l+1))
        match_idx = int(np.searchsorted(r_grid, r_m))
        r_ext = np.concatenate(([0.0], r_grid))
        V_ext = np.concatenate(([0.0], V_eff))  # F(0) wordt niet gebruikt bij u(0)=0
        match_idx_ext = match_idx + 1

        def prof_err(E_arr):
            E_val = float(np.take(E_arr, 0))
            F_ext = 2.0 * (V_ext - E_val)
            
            u0 = 0.0
            u1 = h**(l + 1)
            
            u_f = generalized_numerov_prof_forward(r_ext[:match_idx_ext+2], F_ext[:match_idx_ext+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_ext[match_idx_ext-1:], F_ext[match_idx_ext-1:], 0.0, 1e-10)
            
            u_b = u_b * (np.take(u_f, match_idx_ext) / np.take(u_b, 1))
            
            d_f = (np.take(u_f, match_idx_ext + 1) - np.take(u_f, match_idx_ext - 1)) / (2.0 * h)
            d_b = (np.take(u_b, 2) - np.take(u_b, 0)) / (2.0 * h)
            return [d_f - d_b]

        E_p = float(np.take(fsolve(prof_err, [guess]), 0))
        err_p = abs(E_p - E_th)
        err_prof_list.append(err_p)

        # 3. Bereken de 'Gap' tussen beide methodes
        gap = abs(err_m - err_p)
        direct_diff = abs(E_m - E_p)
        error_gap_list.append(gap)
        direct_diff_list.append(direct_diff)

        print(f"E_{int(E_th - 1.5)} (l={l}, n={n_r}) | {err_m:<12.2e} | {err_p:<12.2e} | {gap:<14.2e} | {direct_diff:<16.2e}")

    # --- PLOT DE GAP TUSSEN BEIDE METHODES ---
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(x, error_gap_list, width=0.5, color='mediumseagreen', edgecolor='black',
                  label=r"Verschil in fout: $||\Delta E_{\mathrm{Matrix}}| - |\Delta E_{\mathrm{Prof}}||$")
    
    # Plot ook het directe verschil |E_Matrix - E_Prof| als lijn erbovenop
    ax.plot(x, direct_diff_list, 'ko--', linewidth=2, markersize=7,
            label=r"Direct verschil tussen methodes: $|E_{\mathrm{Matrix}} - E_{\mathrm{Prof}}|$")

    ax.set_yscale('log')
    ax.set_ylabel("Verschil in energie (log-schaal)", fontsize=11)
    ax.set_title("De 'Gap' tussen Matrix Numerov en Generalized Numerov ($E_0$ t/m $E_5$)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.35)
    ax.legend(fontsize=10)

    # Zet exacte waarden boven de balken
    for bar in bars:
        val = bar.get_height()
        ax.annotate(f"{val:.1e}",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_points = 700
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 6.0, n_points)
    
    l_values = [0, 1]
    alle_energieen = []
    energieen_per_l = {}

    # 1. Bereken energieniveaus voor l = 0, 1, 2, 3 met Matrix-Numerov
    for l in l_values:
        V_eff = 0.5 * r_grid**2 + (l * (l + 1)) / (2.0 * r_grid**2)
        energies, _ = matrix_numerov(r_grid, V_eff)
        
        # Neem de onderste 4 radiale toestanden per l (zolang E <= 7.0)
        valid_E = [E for E in energies[:5] if E < 7.0]
        energieen_per_l[l] = valid_E
        
        # Voeg alle (2l + 1) magnetische subtoestanden (m_l) toe voor de totale ontaarding
        for E in valid_E:
            for m_l in range(-l, l + 1):
                alle_energieen.append(E)

    # 2. Maak een figuur met 2 subplots naast elkaar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- LINKER PLOT: Energieniveaus per l-kanaal (Termschema) ---
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for idx, l in enumerate(l_values):
        for n_r, E in enumerate(energieen_per_l[l]):
            ax1.hlines(E, l - 0.3, l + 0.3, color=colors[idx], linewidth=3)
            ax1.text(l, E + 0.08, f"$n_r={n_r}$\n({E:.2f})", ha='center', va='bottom', fontsize=8.5)

    # Teken stippellijnen voor de exacte theoretische niveaus E = N + 1.5
    for N in range(6):
        E_th = N + 1.5
        ax1.axhline(E_th, color='gray', linestyle=':', alpha=0.4)

    ax1.set_xticks(l_values)
    ax1.set_xticklabels([f"$l = {l}$\n({'spdf'[l]}-golf)" for l in l_values])
    ax1.set_ylabel("Energie $E$ ($\hbar\omega$)", fontsize=11)
    ax1.set_ylim(1.0, 7.2)
    ax1.set_title("Energiespectrum per Impulsmoment $l$", fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.2)

    # --- RECHTER PLOT: Histogram van totale ontaarding (inclusief 2l+1) ---
    bins = np.arange(1.0, 7.5, 1.0)  # Bins gecentreerd op 1.5, 2.5, 3.5, 4.5, 5.5, 6.5
    counts, bin_edges, patches = ax2.hist(alle_energieen, bins=bins, rwidth=0.65, 
                                          color='steelblue', edgecolor='black', alpha=0.85)

    # Zet de exacte getallen (ontaardingsgraad g_N) boven elke balk
    for count, edge in zip(counts, bin_edges[:-1]):
        center = edge + 0.5
        if count > 0:
            ax2.text(center, count + 0.4, f"{int(count)} toestanden", ha='center', fontweight='bold', fontsize=9.5)

    ax2.set_xticks([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    ax2.set_xlabel("Energieniveau $E$ ($\hbar\omega$)", fontsize=11)
    ax2.set_ylabel("Aantal quantumtoestanden (ontaarding)", fontsize=11)
    ax2.set_ylim(0, max(counts) + 3)
    ax2.set_title("Histogram: Totale Ontaarding per Energieniveau", fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_points = 700
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 5.5, n_points)
    # h = r_grid[1] - r_grid[0]
    
    # Defineer de 6 gecombineerde toestanden (E_0 t/m E_5)
    # Format: (label, l, n_r, E_theorie, E_guess, r_match)
    states_info = [
        ("$E_0 = 1.5$\n($l=0, n_r=0$)", 0, 0, 1.5, 1.45, 1.73),
        ("$E_1 = 2.5$\n($l=1, n_r=0$)", 1, 0, 2.5, 2.45, 2.00),
        ("$E_2 = 3.5$\n($l=0, n_r=1$)", 0, 1, 3.5, 3.45, 1.73),
        ("$E_3 = 4.5$\n($l=1, n_r=1$)", 1, 1, 4.5, 4.45, 2.00),
        ("$E_4 = 5.5$\n($l=0, n_r=2$)", 0, 2, 5.5, 5.45, 1.73),
        ("$E_5 = 6.5$\n($l=1, n_r=2$)", 1, 2, 6.5, 6.45, 2.00),
    ]

    # Bereken eerst alle Matrix-Numerov eigenwaarden voor l=0 en l=1
    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)  # l*(l+1)/(2r^2) voor l=1 is 1/r^2
    
    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)

    labels = []
    err_matrix = []
    err_prof = []

    print(f"{'Toestand':<15} | {'E_theorie':<9} | {'Fout Matrix':<12} | {'Fout Prof Gen.':<14}")
    print("-" * 60)

    for label, l, n_r, E_th, guess, r_m in states_info:
        labels.append(label)
        V_eff = V_eff_0 if l == 0 else V_eff_1
        
        # 1. Fout Matrix-Numerov
        E_m = E_mat_l0[n_r] if l == 0 else E_mat_l1[n_r]
        diff_m = abs(E_m - E_th)
        err_matrix.append(diff_m)
        
        # 2. Fout Prof's Generalized Numerov (Shooting)
        match_idx = np.searchsorted(r_grid, r_m)
        def prof_error_func(E_arr):
            E_val = E_arr[0]
            F_arr = 2.0 * (V_eff - E_val)
            u0, u1 = r_grid[0]**(l+1), r_grid[1]**(l+1)
            u_f = generalized_numerov_prof_forward(r_grid[:match_idx+2], F_arr[:match_idx+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_arr[match_idx-1:], 0.0, 1e-10)
            u_b *= (u_f[match_idx] / u_b[1])
            d_f = (u_f[match_idx+1] - u_f[match_idx-1]) / (2.0 * h)
            d_b = (u_b[2] - u_b[0]) / (2.0 * h)
            return [d_f - d_b]

        E_p = fsolve(prof_error_func, [guess])[0]
        diff_p = abs(E_p - E_th)
        err_prof.append(diff_p)
        
        print(f"E_{int(E_th - 1.5)} (l={l}, n={n_r}) | {E_th:<9.1f} | {diff_m:<12.2e} | {diff_p:<14.2e}")

    # --- GROEPS-HISTOGRAM (BAR CHART) ---
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width/2, err_matrix, width, label='Matrix Numerov', color='steelblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, err_prof, width, label="Prof's Generalized Numerov", color='darkorange', edgecolor='black')

    # Logaritmische schaal op de y-as omdat fouten variëren van 10^-11 tot 10^-6
    ax.set_yscale('log')
    ax.set_ylabel(r"Absolute fout $|E_{\mathrm{num}} - E_{\mathrm{theorie}}|$", fontsize=11)
    ax.set_title("Nauwkeurigheid: Matrix Numerov vs. Generalized Numerov ($E_0$ t/m $E_5$)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(1e-13, 1e-3)
    ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.35)
    ax.legend(fontsize=10.5)

    # Zet de exacte wetenschappelijke notatie boven elke balk
    for rect in rects1:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}",
                    xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=0, color='navy')

    for rect in rects2:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}",
                    xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=0, color='darkred')

    plt.tight_layout()
    plt.show()

# =============================================================================
# HOOFDPROGRAMMA VOOR MEERDERE NIVEAUS
# =============================================================================

# We do the exactly the same thing as before, but now for more than just the ground state.
# We also now visualize for the professor's numerov radiale wavefucntions for the first three states.

if __name__ == "__main__":
    n_points = 700  
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 5.5, n_points)
    V_eff = 0.5 * r_grid**2
    match_idx = np.searchsorted(r_grid, 1.73)
    
    # Definieer de proef-startwaarden (guesses) voor E0, E1 en E2
    # Needed only for classic numerov
    guesses = [1.4, 3.4, 5.4]
    # Define the exact energy levels
    theorie = [1.5, 3.5, 5.5]
    
    print(f"\n--- BEREKENING ENERGIENIVEAUS (MEERDERE NIVEAUS) ---")
    
    # 1. Matrix-Numerov (Haalt alles in één keer op!)
    energies_matrix, wf_matrix = matrix_numerov(r_grid, V_eff)
    
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']
    
    for n in range(3):
        E_mat = energies_matrix[n]
        
        # 2 & 3. Shooting loops via fsolve met specifieke guesses
        E_cl = fsolve(matching_error, [guesses[n]], args=(r_grid, V_eff))[0]
        E_pr = fsolve(prof_matching_error, [guesses[n]], args=(r_grid, V_eff))[0]
        
        print(f"\nToestand n = {n}:")
        print(f"  Matrix-Numerov: E = {E_mat:.5f} | Fout: {abs(E_mat - theorie[n]):.2e}")
        print(f"  Klassiek Shoot: E = {E_cl:.5f} | Fout: {abs(E_cl - theorie[n]):.2e}")
        print(f"  Prof Methode  : E = {E_pr:.5f} | Fout: {abs(E_pr - theorie[n]):.2e}")
        
        # Golffunctie van de Prof genereren voor deze specifieke toestand
        F_opt = 2.0 * (V_eff - E_pr)
        u_f = generalized_numerov_prof_forward(r_grid[:match_idx+1], F_opt[:match_idx+1], 0.0, 1e-5)
        u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_opt[match_idx-1:], 0.0, 1e-10)
        u_b *= (u_f[-1] / u_b[1])
        
        u_shoot = np.concatenate([u_f[:-1], u_b[1:]])
        u_shoot /= np.max(np.abs(u_shoot)) # Normaliseren op amplitude
        
        # Plot de golffunctie
        plt.plot(r_grid, u_shoot, label=f"Toestand $E_{n}$ (Prof Numerov = {E_pr:.3f})", color=colors[n], linewidth=2)

    plt.title("Radiale Golffuncties $u(r)$ voor $E_0, E_1$ en $E_2$ (Prof's Numerov)")
    plt.xlabel("Radiale afstand $r$")
    plt.ylabel("$u(r)$")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# HOOFDPROGRAMMA EN 2x2 PLOT LOGICA
# =============================================================================

# Now, finally, we visualize the three radiale wavefucntiosn for all three methods.
# On top if that, we make a histogram with the absolute difference against the theoretical values, for each method.
# We show all this in a 2x2 grid.

if __name__ == "__main__":
    n_points = 700
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 5.5, n_points)
    V_eff = 0.5 * r_grid**2
    match_idx = np.searchsorted(r_grid, 1.73)
    
    guesses = [1.4, 3.4, 5.4]
    theorie = [1.5, 3.5, 5.5]
    
    # Bereken exact analytisch voor normalisatie-referentie
    u_exact = [
        r_grid * np.exp(-r_grid**2 / 2.0),
        r_grid * (3.0 - 2.0 * r_grid**2) * np.exp(-r_grid**2 / 2.0),
        r_grid * (15.0 - 20.0 * r_grid**2 + 4.0 * r_grid**4) * np.exp(-r_grid**2 / 2.0)
    ]
    for n in range(3):
        u_exact[n] /= np.max(np.abs(u_exact[n]))

    # 1. Berekening Matrix-Numerov
    energies_matrix, wf_matrix = matrix_numerov(r_grid, V_eff)

    # Initialiseer de 2x2 figuur
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    colors = ['b', 'g', 'r']
    
    # Dataopslag voor foutenanalyse (4e grafiek)
    error_matrix = np.zeros(3)
    error_classical = np.zeros(3)
    error_prof = np.zeros(3)

    for n in range(3):
        # Bereken exacte numerieke eigenwaarden per niveau
        E_mat = energies_matrix[n]
        E_cl = fsolve(matching_error, [guesses[n]], args=(r_grid, V_eff))[0]
        E_pr = fsolve(prof_matching_error, [guesses[n]], args=(r_grid, V_eff))[0]
        
        error_matrix[n] = abs(E_mat - theorie[n])
        error_classical[n] = abs(E_cl - theorie[n])
        error_prof[n] = abs(E_pr - theorie[n])
        
        # --- SUBPLOT 1: MATRIX-NUMEROV ---
        u_m = wf_matrix[:, n]
        u_m /= np.max(np.abs(u_m))
        if np.sum(u_m * u_exact[n]) < 0: u_m = -u_m  # Spiegel-fix
        axs[0, 0].plot(r_grid, u_m, color=colors[n], linestyle='-', linewidth=2, label=f"$E_{n}$ = {E_mat:.4f}")
        
        # --- SUBPLOT 2: CLASSICAL SHOOTING-NUMEROV ---
        k2_opt = 2.0 * (E_cl - V_eff)
        u_f_cl = classical_numerov_forward(r_grid[:match_idx+1], k2_opt[:match_idx+1], 0.0, 1e-5)
        u_b_cl = classical_numerov_backward(r_grid[match_idx-1:], k2_opt[match_idx-1:], 0.0, 1e-10)
        u_b_cl *= (u_f_cl[-1] / u_b_cl[1])
        u_shoot_cl = np.concatenate([u_f_cl[:-1], u_b_cl[1:]])
        u_shoot_cl /= np.max(np.abs(u_shoot_cl))
        if np.sum(u_shoot_cl * u_exact[n]) < 0: u_shoot_cl = -u_shoot_cl
        axs[0, 1].plot(r_grid, u_shoot_cl, color=colors[n], linestyle='-', linewidth=2, label=f"$E_{n}$ = {E_cl:.4f}")

        # --- SUBPLOT 3: GENERALIZED (PROF) NUMEROV ---
        F_opt = 2.0 * (V_eff - E_pr)
        u_f_pr = generalized_numerov_prof_forward(r_grid[:match_idx+1], F_opt[:match_idx+1], 0.0, 1e-5)
        u_b_pr = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_opt[match_idx-1:], 0.0, 1e-10)
        u_b_pr *= (u_f_pr[-1] / u_b_pr[1])
        u_shoot_pr = np.concatenate([u_f_pr[:-1], u_b_pr[1:]])
        u_shoot_pr /= np.max(np.abs(u_shoot_pr))
        if np.sum(u_shoot_pr * u_exact[n]) < 0: u_shoot_pr = -u_shoot_pr
        axs[1, 0].plot(r_grid, u_shoot_pr, color=colors[n], linestyle='-', linewidth=2, label=f"$E_{n}$ = {E_pr:.4f}")

    # Voeg analytische referentielijnen toe aan subplots 1, 2 en 3
    for n in range(3):
        axs[0, 0].plot(r_grid, u_exact[n], 'k--', alpha=0.35, linewidth=1.5, label="Analytisch" if n==0 else "")
        axs[0, 1].plot(r_grid, u_exact[n], 'k--', alpha=0.35, linewidth=1.5, label="Analytisch" if n==0 else "")
        axs[1, 0].plot(r_grid, u_exact[n], 'k--', alpha=0.35, linewidth=1.5, label="Analytisch" if n==0 else "")

    # Titels en lay-out voor de golffuncties
    methods = [
        (axs[0, 0], "1. Matrix-Numerov (Diagonalisatie)"),
        (axs[0, 1], "2. Classical Shooting-Numerov"),
        (axs[1, 0], "3. Prof's Generalized Numerov")
    ]
    for ax, title in methods:
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Straal $r$")
        ax.set_ylabel("$u(r)$")
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    # --- SUBPLOT 4: BENCHMARK ENERGIEFOUT ---
    x_indices = np.arange(3)
    bar_width = 0.25
    axs[1, 1].bar(x_indices - bar_width, error_matrix, bar_width, label="Matrix-Numerov", color='royalblue')
    axs[1, 1].bar(x_indices, error_classical, bar_width, label="Klassieke Shooting", color='forestgreen')
    axs[1, 1].bar(x_indices + bar_width, error_prof, bar_width, label="Prof's Numerov", color='crimson')
    
    axs[1, 1].set_title("4. Absolute Energiefout $|E_{num} - E_{exact}|$", fontsize=12, fontweight='bold')
    axs[1, 1].set_xticks(x_indices)
    axs[1, 1].set_xticklabels(["$E_0$ (1.5)", "$E_1$ (3.5)", "$E_2$ (5.5)"])
    axs[1, 1].set_ylabel("Fout (Logaritmische schaal)")
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True, which="both", alpha=0.3)
    axs[1, 1].legend(loc="best")

    plt.suptitle("3D Harmonische Oscillator ($l=0$) - Algoritme Vergelijking", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    n_points = 700
    
    n_points = 700
    r_max = 5.5
    h = r_max / n_points

    # Start bij h en eindig bij r_max (nu geldt exact: r_grid[0] == h)
    r_grid = np.linspace(h, r_max, n_points)

    # r_grid = np.linspace(1e-4, 5.5, n_points)
    V_eff = 0.5 * r_grid**2
    match_idx = np.searchsorted(r_grid, 1.73)
    
    guesses = [1.4, 3.4, 5.4]
    theorie = [1.5, 3.5, 5.5]
    
    # Bereken exact analytisch voor normalisatie-referentie
    u_exact = [
        r_grid * np.exp(-r_grid**2 / 2.0),
        r_grid * (3.0 - 2.0 * r_grid**2) * np.exp(-r_grid**2 / 2.0),
        r_grid * (15.0 - 20.0 * r_grid**2 + 4.0 * r_grid**4) * np.exp(-r_grid**2 / 2.0)
    ]
    for n in range(3):
        u_exact[n] /= np.max(np.abs(u_exact[n]))

    # 1. Berekening Matrix-Numerov
    energies_matrix, wf_matrix = matrix_numerov(r_grid, V_eff)
    colors = ['b', 'g', 'r']
    
    error_matrix = np.zeros(3)
    error_prof = np.zeros(3)
    
    # Opslag voor golffuncties van de prof om te kunnen plotten
    u_prof_list = []

    for n in range(3):
        E_mat = energies_matrix[n]
        E_pr = fsolve(prof_matching_error, [guesses[n]], args=(r_grid, V_eff))[0]
        
        error_matrix[n] = abs(E_mat - theorie[n])
        error_prof[n] = abs(E_pr - theorie[n])
        
        # Prof's Numerov Golffunctie berekenen
        F_opt = 2.0 * (V_eff - E_pr)
        u_f_pr = generalized_numerov_prof_forward(r_grid[:match_idx+1], F_opt[:match_idx+1], 0.0, 1e-5)
        u_b_pr = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_opt[match_idx-1:], 0.0, 1e-10)
        u_b_pr *= (u_f_pr[-1] / u_b_pr[1])
        u_shoot_pr = np.concatenate([u_f_pr[:-1], u_b_pr[1:]])
        u_shoot_pr /= np.max(np.abs(u_shoot_pr))
        if np.sum(u_shoot_pr * u_exact[n]) < 0: u_shoot_pr = -u_shoot_pr
        u_prof_list.append((u_shoot_pr, E_pr))

    # =========================================================================
    # FIGUUR 1: EÃ©n grote figuur met 2 subplots (Prof Numerov & Matrix Numerov)
    # =========================================================================
    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 5))
    
    for n in range(3):
        # Subplot 1: Prof's Numerov
        u_pr, E_pr = u_prof_list[n]
        axs1[0].plot(r_grid, u_pr, color=colors[n], linestyle='-', linewidth=2, label=f"$E_{n}$ = {E_pr:.4f}")
        
        # Subplot 2: Matrix Numerov
        u_m = wf_matrix[:, n]
        u_m /= np.max(np.abs(u_m))
        if np.sum(u_m * u_exact[n]) < 0: u_m = -u_m
        axs1[1].plot(r_grid, u_m, color=colors[n], linestyle='-', linewidth=2, label=f"$E_{n}$ = {energies_matrix[n]:.4f}")

    # Analytische referentielijnen toevoegen aan beide subplots
    for n in range(3):
        axs1[0].plot(r_grid, u_exact[n], 'k--', alpha=0.35, linewidth=1.5, label="Analytisch" if n==0 else "")
        axs1[1].plot(r_grid, u_exact[n], 'k--', alpha=0.35, linewidth=1.5, label="Analytisch" if n==0 else "")

    # Layout Subplot 1 (Prof Numerov)
    axs1[0].set_title("1. Prof's Gegeneraliseerde Numerov", fontsize=12, fontweight='bold')
    axs1[0].set_xlabel("Straal $r$")
    axs1[0].set_ylabel("$u(r)$")
    axs1[0].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axs1[0].grid(True, alpha=0.3)
    axs1[0].legend(loc="upper right")

    # Layout Subplot 2 (Matrix Numerov)
    axs1[1].set_title("2. Matrix-Numerov (Diagonalisatie)", fontsize=12, fontweight='bold')
    axs1[1].set_xlabel("Straal $r$")
    axs1[1].set_ylabel("$u(r)$")
    axs1[1].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axs1[1].grid(True, alpha=0.3)
    axs1[1].legend(loc="upper right")

    plt.suptitle("Vergelijking Radiale Golffuncties ($l=0$)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # FIGUUR 2: Een afzonderlijke plot voor de absolute energiefouten
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    
    x_indices = np.arange(3)
    bar_width = 0.35
    ax2.bar(x_indices - bar_width/2, error_matrix, bar_width, label="Matrix-Numerov", color='royalblue')
    ax2.bar(x_indices + bar_width/2, error_prof, bar_width, label="Prof's Numerov", color='crimson')
    
    ax2.set_title("Absolute Energiefout $|E_{num} - E_{exact}|$", fontsize=12, fontweight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(["$E_0$ (1.5)", "$E_1$ (3.5)", "$E_2$ (5.5)"])
    ax2.set_ylabel("Fout (Logaritmische schaal)")
    ax2.set_yscale('log')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")
    
    plt.tight_layout()
    plt.show()




import numpy as np
import scipy.linalg as la
from itertools import product



# Matrix Numerov (Cartesisch 1D)
#################################
def matrix_numerov_cartesian_1d(x_grid, V_1d):
    n = len(x_grid)
    h = x_grid[1] - x_grid[0]
    num_internal = n - 2
    
    A = np.zeros((num_internal, num_internal))
    np.fill_diagonal(A, -2.0)
    np.fill_diagonal(A[1:], 1.0)
    np.fill_diagonal(A[:, 1:], 1.0)
    A = A / (h**2)
    
    B = np.zeros((num_internal, num_internal))
    np.fill_diagonal(B, 10.0 / 12.0)
    np.fill_diagonal(B[1:], 1.0 / 12.0)
    np.fill_diagonal(B[:, 1:], 1.0 / 12.0)
    
    H = -0.5 * (la.inv(B) @ A) + np.diag(V_1d[1:-1])
    H = 0.5 * (H + H.T)
    
    energies_1d, wf_internal = la.eigh(H)
    
    wf_1d = np.zeros((n, num_internal))
    wf_1d[1:-1, :] = wf_internal
    
    return energies_1d, wf_1d


# =============================================================================
# HOOFDPROGRAMMA: TAAK 1 (VERGELIJKING E_0 t/m E_5 VOOR 3 METHODES)
# =============================================================================

if __name__ == "__main__":
    n_points = 700
    r_max = 5.5
    h = r_max / n_points
    r_grid = np.linspace(h, r_max, n_points)

    # 1. Berekening voor Sferische / BolcoÃ¶rdinaten (Matrix & Prof)
    states_info = [
        ("$E_0 = 1.5$\n($l=0, n_r=0$)", 0, 0, 1.5, 1.48, 1.73, 0),
        ("$E_1 = 2.5$\n($l=1, n_r=0$)", 1, 0, 2.5, 2.48, 2.00, 1),
        ("$E_2 = 3.5$\n($l=0, n_r=1$)", 0, 1, 3.5, 3.48, 1.73, 2),
        ("$E_3 = 4.5$\n($l=1, n_r=1$)", 1, 1, 4.5, 4.48, 2.00, 3),
        ("$E_4 = 5.5$\n($l=0, n_r=2$)", 0, 2, 5.5, 5.48, 1.73, 4),
        ("$E_5 = 6.5$\n($l=1, n_r=2$)", 1, 2, 6.5, 6.48, 2.00, 5),
    ]

    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)
    
    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)

    # 2. Berekening voor Cartesische coÃ¶rdinaten (1D rooster & combinaties)
    x_max = 5.5
    x_cart_grid = np.linspace(-x_max, x_max, n_points)
    V_1d = 0.5 * x_cart_grid**2
    E_1d, _ = matrix_numerov_cartesian_1d(x_cart_grid, V_1d)

    labels = []
    err_matrix_sph = []
    err_matrix_cart = []
    err_prof = []

    print(f"\n{'Toestand':<15} | {'E_theorie':<9} | {'Fout Mat. Sph':<13} | {'Fout Mat. Cart':<14} | {'Fout Prof Gen.':<14}")
    print("-" * 75)

    for label, l, n_r, E_th, guess, r_m, cart_n in states_info:
        labels.append(label)
        V_eff = V_eff_0 if l == 0 else V_eff_1
        
        # A. Matrix-Numerov (Sferisch)
        E_m_sph = E_mat_l0[n_r] if l == 0 else E_mat_l1[n_r]
        diff_m_sph = abs(E_m_sph - E_th)
        err_matrix_sph.append(diff_m_sph)
        
        # B. Matrix-Numerov (Cartesisch - isotrope 3D oscillator via (N, 0, 0))
        E_m_cart = E_1d[cart_n] + 2.0 * E_1d[0]
        diff_m_cart = abs(E_m_cart - E_th)
        err_matrix_cart.append(diff_m_cart)
        
        # C. Prof's Generalized Numerov (Shooting)
        match_idx = np.searchsorted(r_grid, r_m)
        def prof_error_func(E_arr):
            E_val = E_arr[0]
            F_arr = 2.0 * (V_eff - E_val)
            u0, u1 = r_grid[0]**(l+1), r_grid[1]**(l+1)
            u_f = generalized_numerov_prof_forward(r_grid[:match_idx+2], F_arr[:match_idx+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_arr[match_idx-1:], 0.0, 1e-10)
            u_b *= (u_f[match_idx] / u_b[1])
            d_f = (u_f[match_idx+1] - u_f[match_idx-1]) / (2.0 * h)
            d_b = (u_b[2] - u_b[0]) / (2.0 * h)
            return [d_f - d_b]

        E_p = fsolve(prof_error_func, [guess])[0]
        diff_p = abs(E_p - E_th)
        err_prof.append(diff_p)
        
        print(f"E_{int(E_th - 1.5)} (l={l}, n={n_r}) | {E_th:<9.1f} | {diff_m_sph:<13.2e} | {diff_m_cart:<14.2e} | {diff_p:<14.2e}")

    # --- HISTOGRAM / STAAFDIAGRAM MET DRIE METHODES ---
    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, err_matrix_sph, width, label='Matrix Numerov (Sferisch)', color='steelblue', edgecolor='black')
    rects2 = ax.bar(x, err_matrix_cart, width, label='Matrix Numerov (Cartesisch)', color='forestgreen', edgecolor='black')
    rects3 = ax.bar(x + width, err_prof, width, label="Prof's Generalized Numerov", color='darkorange', edgecolor='black')

    ax.set_yscale('log')
    ax.set_ylabel(r"Absolute fout $|E_{\mathrm{num}} - E_{\mathrm{theorie}}|$", fontsize=11)
    ax.set_title("Nauwkeurigheid: Matrix (Sferisch) vs. Matrix (Cartesisch) vs. Prof Numerov ($E_0$ t/m $E_5$)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(1e-13, 1e-3)
    ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.35)
    ax.legend(fontsize=10)

    # Waarden boven de balken plaatsen
    for rect in rects1:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}", (rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, color='navy')

    for rect in rects2:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}", (rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, color='darkgreen')

    for rect in rects3:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}", (rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, color='darkred')

    plt.tight_layout()
    plt.show()



import numpy as np
import scipy.linalg as la

# =============================================================================
# MATRIX-NUMEROV PRINT SCRIPT (TAAK 1)
# =============================================================================

def print_matrix_numerov_results(r_grid, x_cart_grid):
    n_points = len(r_grid)
    h = r_grid[1] - r_grid[0]
    
    # 1. Potentiaal en Matrix-Numerov (Sferisch / Bolcoördinaten)
    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)  # l=1 centrifugale term l(l+1)/(2r^2)
    
    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)
    
    # 2. Potentiaal en Matrix-Numerov (Cartesisch 1D)
    V_1d = 0.5 * x_cart_grid**2
    E_1d, _ = matrix_numerov_cartesian_1d(x_cart_grid, V_1d)
    
    # Definiëreeer de 6 toestanden: (Label, l, n_r, E_theorie, Cartesische index nx)
    states_info = [
        ("E0 (l=0, nr=0)", 0, 0, 1.5, 0),
        ("E1 (l=1, nr=0)", 1, 0, 2.5, 1),
        ("E2 (l=0, nr=1)", 0, 1, 3.5, 2),
        ("E3 (l=1, nr=1)", 1, 1, 4.5, 3),
        ("E4 (l=0, nr=2)", 0, 2, 5.5, 4),
        ("E5 (l=1, nr=2)", 1, 2, 6.5, 5),
    ]
    
    print("\n" + "="*85)
    print(f"{'TOESTAND':<16} | {'THEORIE':<9} | {'MAT. SFERISCH':<14} | {'FAUT (SPHER)':<13} | {'MAT. CARTES':<13} | {'FAUT (CART)':<12}")
    print("="*85)
    
    for label, l, n_r, E_th, cart_n in states_info:
        # Sferische Matrix-Numerov waarde
        E_sph = E_mat_l0[n_r] if l == 0 else E_mat_l1[n_r]
        fout_sph = abs(E_sph - E_th)
        
        # Cartesische Matrix-Numerov waarde (isotrope 3D oscillator via E_nx + 2*E_0)
        E_cart = E_1d[cart_n] + 2.0 * E_1d[0]
        fout_cart = abs(E_cart - E_th)
        
        print(f"{label:<16} | {E_th:<9.1f} | {E_sph:<14.6f} | {fout_sph:<13.2e} | {E_cart:<13.6f} | {fout_cart:<12.2e}")
        
    print("="*85 + "\n")

if __name__ == "__main__":
    # Roosterinstellingen
    n_points = 700
    r_max = 5.5
    h = r_max / n_points
    r_grid = np.linspace(h, r_max, n_points)
    
    x_max = 5.5
    x_cart_grid = np.linspace(-x_max, x_max, n_points)
    
    # Aanroepen van de print-functie
    # (Zorg dat matrix_numerov en matrix_numerov_cartesian_1d hierboven gedefinieerd zijn)
    # print_matrix_numerov_results(r_grid, x_cart_grid)



# =============================================================================
# HOOFDPROGRAMMA: TABEL MET ENERGIEËN EN ABSOLUTE FOUTEN
# =============================================================================

if __name__ == "__main__":
    n_points = 700
    r_max = 5.5
    h = r_max / n_points
    r_grid = np.linspace(h, r_max, n_points)
    
    x_max = 5.5
    x_cart_grid = np.linspace(-x_max, x_max, n_points)
    
    # Effectieve potentialen voor sferisch en Cartesisch
    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)
    V_1d = 0.5 * x_cart_grid**2
    
    # Bereken sferische en Cartesische matrix eigenwaarden
    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)
    E_1d, _ = matrix_numerov_cartesian_1d(x_cart_grid, V_1d)
    
    # Specificatie van E_0 t/m E_5: (Label, l, n_r, E_theorie, guess, r_match, cart_idx)
    states_info = [
        ("E0 (l=0, nr=0)", 0, 0, 1.5, 1.48, 1.73, 0),
        ("E1 (l=1, nr=0)", 1, 0, 2.5, 2.48, 2.00, 1),
        ("E2 (l=0, nr=1)", 0, 1, 3.5, 3.48, 1.73, 2),
        ("E3 (l=1, nr=1)", 1, 1, 4.5, 4.48, 2.00, 3),
        ("E4 (l=0, nr=2)", 0, 2, 5.5, 5.48, 1.73, 4),
        ("E5 (l=1, nr=2)", 1, 2, 6.5, 6.48, 2.00, 5),
    ]
    
    print("\n" + "="*115)
    print(f"{'Toestand':<16} | {'Theorie':<8} | {'Mat. Sferisch':<13} | {'|dE| Sferisch':<13} | {'Mat. Cartes':<13} | {'|dE| Cartes':<13} | {'Prof Gen.':<12} | {'|dE| Prof':<11}")
    print("="*115)
    
    for label, l, n_r, E_th, guess, r_m, cart_n in states_info:
        # 1. Matrix-Numerov (Sferisch)
        E_sph = E_mat_l0[n_r] if l == 0 else E_mat_l1[n_r]
        err_sph = abs(E_sph - E_th)
        
        # 2. Matrix-Numerov (Cartesisch 3D via 1D sommatie)
        E_cart = E_1d[cart_n] + 2.0 * E_1d[0]
        err_cart = abs(E_cart - E_th)
        
        # 3. Prof's Generalized Shooting Numerov
        V_eff = V_eff_0 if l == 0 else V_eff_1
        match_idx = np.searchsorted(r_grid, r_m)
        
        def prof_err(E_arr):
            E_val = E_arr[0]
            F_arr = 2.0 * (V_eff - E_val)
            u0, u1 = r_grid[0]**(l + 1), r_grid[1]**(l + 1)
            u_f = generalized_numerov_prof_forward(r_grid[:match_idx+2], F_arr[:match_idx+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_arr[match_idx-1:], 0.0, 1e-10)
            u_b *= (u_f[match_idx] / u_b[1])
            d_f = (u_f[match_idx+1] - u_f[match_idx-1]) / (2.0 * h)
            d_b = (u_b[2] - u_b[0]) / (2.0 * h)
            return [d_f - d_b]

        E_prof = fsolve(prof_err, [guess])[0]
        err_prof = abs(E_prof - E_th)
        
        print(f"{label:<16} | {E_th:<8.1f} | {E_sph:<13.6f} | {err_sph:<13.2e} | {E_cart:<13.6f} | {err_cart:<13.2e} | {E_prof:<12.6f} | {err_prof:<11.2e}")
        
    print("="*115 + "\n")

# That was task 1 # comment here
# Now task 2
##################################################################################################################################################################################




import numpy as np
import scipy.special as sp
import scipy.signal as signal
import matplotlib.pyplot as plt

# =============================================================================
# TAAK 2: H-Kr VERSTROOIING & FASEVERSCHUIVINGEN PRINTEN
# =============================================================================

EPSILON = 5.9          # meV
SIGMA = 1.0            # Eenheid van lengte = 1 sigma (3.75 Å)
FACTOR_2M_HBAR2 = 6.12 # meV^-1 * sigma^-2
CONV_ANG_SQ = 3.75**2  # Omrekenfactor naar Å^2

def lennard_jones_potential(r):
    r_safe = np.maximum(r, 1e-8)
    term = SIGMA / r_safe
    return EPSILON * (term**12 - 2.0 * term**6)

def F_scattering(l, r, E):
    """Voor de gegeneraliseerde Numerov (u'' = F * u)"""
    V_lj = lennard_jones_potential(r)
    V_cent = (l * (l + 1.0)) / (FACTOR_2M_HBAR2 * r**2)
    return FACTOR_2M_HBAR2 * (V_lj + V_cent - E)

def calculate_phase_shift(l, E, r_grid, u_sol):
    """Berekent de faseverschuiving delta_l via vergelijking (8) en (9)."""
    k = np.sqrt(FACTOR_2M_HBAR2 * E)
    idx1, idx2 = -50, -1
    r1, r2 = r_grid[idx1], r_grid[idx2]
    u1, u2 = u_sol[idx1], u_sol[idx2]
    
    K = (r1 * u2) / (r2 * u1)
    
    jl_1 = sp.spherical_jn(l, k * r1)
    jl_2 = sp.spherical_jn(l, k * r2)
    nl_1 = sp.spherical_yn(l, k * r1)
    nl_2 = sp.spherical_yn(l, k * r2)
    
    tan_delta = (K * jl_1 - jl_2) / (K * nl_1 - nl_2)
    return np.arctan(tan_delta)

def compute_cross_section_method_with_printing(method_type, E_range, l_max=6, r_min=0.5, r_max=15.0, n_points=3000):
    """
    Berekent sigma_tot en print de faseverschuivingen delta_l op het scherm.
    """
    r_grid = np.linspace(r_min, r_max, n_points)
    
    const_asymp = np.sqrt(FACTOR_2M_HBAR2 * EPSILON / 25.0)
    u0 = np.exp(-const_asymp * (r_grid[0]**(-5)))
    u1 = np.exp(-const_asymp * (r_grid[1]**(-5)))
    
    sigma_tot_list = []
    partial_sigma = np.zeros((l_max + 1, len(E_range)))
    
    print(f"\n==================================================")
    print(f" RESULTATEN FASEVERSCHUIVINGEN: {method_type.upper()} NUMEROV")
    print(f"==================================================")
    print(f"{'E (meV)':<10} | {'l':<5} | {'Delta_l (rad)':<15} | {'Delta_l (deg)':<15}")
    print("-" * 55)
    
    for idx_E, E in enumerate(E_range):
        k2_val = FACTOR_2M_HBAR2 * E
        sum_cross_section = 0.0
        
        for l in range(l_max + 1):
            F_arr = F_scattering(l, r_grid, E)
            
            if method_type == 'generalized':
                u_sol = generalized_numerov_prof_forward(r_grid, F_arr, u0, u1)
            elif method_type == 'classical':
                k2_arr = -F_arr
                u_sol = classical_numerov_forward(r_grid, k2_arr, u0, u1)
            else:
                raise ValueError("Onbekende methode type")
                
            delta_l = calculate_phase_shift(l, E, r_grid, u_sol)
            
            # Print de faseverschuiving (om de console niet te overspoelen printen we hier bijvoorbeeld elke 50e stap)
            if idx_E % 100 == 0:
                print(f"{E:<10.3f} | {l:<5} | {delta_l:<15.5f} | {np.degrees(delta_l):<15.2f}")
            
            sigma_l = (4.0 * np.pi / k2_val) * (2.0 * l + 1.0) * (np.sin(delta_l)**2)
            partial_sigma[l, idx_E] = sigma_l * CONV_ANG_SQ
            sum_cross_section += sigma_l
            
        sigma_tot_list.append(sum_cross_section * CONV_ANG_SQ)
        
    return r_grid, E_range, np.array(sigma_tot_list), partial_sigma

from scipy.integrate import quad

def born_scattering_cross_section(E_range):
    """
    Berekent de totale werkzame doorsnede via de Eerste Born-benadering 
    voor de Lennard-Jones potentiaal.
    """
    sigma_born_list = []
    k_vals = np.sqrt(FACTOR_2M_HBAR2 * E_range)
    
    for k in k_vals:
        # Integrand voor de verstrooiingsamplitude als functie van q (of integratie over r)
        # Voor s-waarde / totale doorsnede in Born benadering via centrale potentiaal:
        def integrand(r):
            V = lennard_jones_potential(r)
            # Integrum voor de s-golf Born faseverschuiving of amplitude
            # δ_l^Born = -k ∫_0^inf V(r) [j_l(kr)]^2 r^2 dr (voor l=0)
            return V * (np.sin(k * r) / (k * r))**2 * r**2

        # Integratie over de straal r (van r_min tot grote waarde)
        integral, _ = quad(integrand, 0.5, 50.0)
        
        # Eerste orde Born benadering voor s-golf faseverschuiving: delta_0 ≈ -k ∫ V r^2 dr / ...
        delta_born = -FACTOR_2M_HBAR2 * integral
        
        # Totale doorsnede benadering voor s-wgolf (sigma = (4pi/k^2) * sin^2(delta))
        sigma_l0 = (4.0 * np.pi / (k**2)) * (np.sin(delta_born)**2)
        sigma_born_list.append(sigma_l0 * CONV_ANG_SQ)
        
    return np.array(sigma_born_list)



from scipy.integrate import quad

def born_scattering_cross_section_multiwav(E_range, l_max=6):
    """
    Berekent de totale werkzame doorsnede via de Eerste Born-benadering 
    voor alle partiële golven l tot l_max.
    """
    sigma_born_tot_list = []
    
    for E in E_range:
        k = np.sqrt(FACTOR_2M_HBAR2 * E)
        k2_val = FACTOR_2M_HBAR2 * E
        sum_sigma_born = 0.0
        
        for l in range(l_max + 1):
            # Integrand voor partiële golf l in de Born-benadering:
            # delta_l^Born = -k * 2m/hbar^2 * int V(r) [j_l(kr)]^2 r^2 dr
            def integrand(r):
                V = lennard_jones_potential(r)
                # sferische Bessel-functie van orde l
                jl = sp.spherical_jn(l, k * r)
                return V * (jl**2) * (r**2)

            # Numerieke integratie van r = 0.5 tot r = 15.0
            integral, _ = quad(integrand, 0.5, 15.0, limit=100)
            
            # Eerste orde Born faseverschuiving voor deze l
            delta_born_l = -FACTOR_2M_HBAR2 * k * integral
            
            # Partiële doorsnede voor deze l in Born benadering
            sigma_l_born = (4.0 * np.pi / k2_val) * (2.0 * l + 1.0) * (np.sin(delta_born_l)**2)
            sum_sigma_born += sigma_l_born
            
        sigma_born_tot_list.append(sum_sigma_born * CONV_ANG_SQ)
        
    return np.array(sigma_born_tot_list)

# =============================================================================
# HOOFDPROGRAMMA
# =============================================================================
if __name__ == "__main__":
    # Kleinere energieroosterstap om de printlijnen overzichtelijk te houden
    E_vals = np.linspace(0.1, 3.5, 300)
    l_max = 6 # Beperkt tot l_max = 6 zodat de print tabel overzichtelijk blijft
    
    # 1. Gegeneraliseerde Numerov
    _, _, sigma_gen, partial_gen = compute_cross_section_method_with_printing('generalized', E_vals, l_max=l_max)
    
    # 2. Klassieke Numerov
    _, _, sigma_cls, partial_cls = compute_cross_section_method_with_printing('classical', E_vals, l_max=l_max)
    
    # --- PLOT 1: GEGENERALISEERD ---
    plt.figure(figsize=(10, 6))
    plt.plot(E_vals, sigma_gen, 'k-', linewidth=2.5, label=r"$\sigma_{\text{tot}}$")
    # Bereken de Born doorsnede
    # 2. Born-benadering berekenen (voor alle l tot l_max)
    sigma_born = born_scattering_cross_section_multiwav(E_vals, l_max=l_max)

    plt.plot(E_vals, sigma_born, 'm:', linewidth=2.0, label="Born")
    
    # 2. Vind de pieken met scipy.signal.find_peaks
    # 'prominence' filtert kleine numerieke rimpelingen weg
    peaks_gen, _ = signal.find_peaks(sigma_gen, prominence=5.0)
    peaks_born, _ = signal.find_peaks(sigma_born, prominence=5.0)

    # 3. Plot de pieken op de grafiek
    # plt.plot(E_vals[peaks_gen], sigma_gen[peaks_gen], 'ro', markersize=8, zorder=5, label="Pieken Numerov")
    # plt.plot(E_vals[peaks_born], sigma_born[peaks_born], 'm^', markersize=8, zorder=5, label="Pieken Born")

    # 4. Voeg tekstlabels met de energiewaarde toe boven elke piek
    # for idx in peaks_gen:
      #  plt.annotate(f"{E_vals[idx]:.2f} meV", 
           #          (E_vals[idx], sigma_gen[idx]),
           #          textcoords="offset points", xytext=(0, 8), ha='center', 
           #          fontsize=9, fontweight='bold', color='red')

    #for idx in peaks_born:
     #   plt.annotate(f"{E_vals[idx]:.2f} meV", 
                 #    (E_vals[idx], sigma_born[idx]),
                 #    textcoords="offset points", xytext=(0, 8), ha='center', 
                 #    fontsize=9, color='purple')

    # 5. Print de exacte piekwaarden in de console
    print("\n--- ALLE GEVONDEN PIEKEN (NUMEROV) ---")
    for idx in peaks_gen:
        print(f"E = {E_vals[idx]:.4f} meV | sigma = {sigma_gen[idx]:.2f} Å²")

    print("\n--- ALLE GEVONDEN PIEKEN (BORN) ---")
    for idx in peaks_born:
        print(f"E = {E_vals[idx]:.4f} meV | sigma = {sigma_born[idx]:.2f} Å²")

    # Selecteer enkel de 3 HOOGSTE pieken (gesorteerd op sigma-waarde)
    top3_gen = peaks_gen[np.argsort(sigma_gen[peaks_gen])[-3:]]
    top3_born = peaks_born[np.argsort(sigma_born[peaks_born])[-3:]]

    # 3. Plot enkel de top 3 pieken op de grafiek
    plt.plot(E_vals[top3_gen], sigma_gen[top3_gen], 'ro', markersize=8, zorder=5, label="Top 3 Pieken Numerov")
    plt.plot(E_vals[top3_born], sigma_born[top3_born], 'm^', markersize=8, zorder=5, label="Top 3 Pieken Born")
    
    # --- FWHM (PIEKBREEDTE / LEVENSDUUR) BEREKENEN EN VISUALISEREN ---
    print("\n--- PIEKBREEDTES (FWHM / GAMMA) VOOR TOP 3 PIEKEN ---")
    for idx in top3_gen:
        peak_sigma = sigma_gen[idx]
        half_max = peak_sigma / 2.0
        
        # Zoek de linker en rechter snijpunten op halve hoogte
        left_side = sigma_gen[:idx]
        E_left_grid = E_vals[:idx]
        idx_l = np.argmin(np.abs(left_side - half_max)) if len(left_side) > 0 else idx
        E1 = E_left_grid[idx_l]
        
        right_side = sigma_gen[idx:]
        E_right_grid = E_vals[idx:]
        idx_r = np.argmin(np.abs(right_side - half_max)) if len(right_side) > 0 else 0
        E2 = E_right_grid[idx_r]
        
        gamma = E2 - E1
        print(f"Piek bij E = {E_vals[idx]:.4f} meV -> Breedte (Gamma) = {gamma:.4f} meV")
        
        # Teken een horizontale lijn op de halve hoogte om de breedte aan te duiden
        plt.hlines(half_max, E1, E2, color='darkred', linestyle='--', linewidth=1.5, zorder=4)
        plt.annotate(f"$\\Gamma = {gamma:.3f}$ meV", 
                     (E_vals[idx], half_max),
                     textcoords="offset points", xytext=(0, -15), ha='center', 
                     fontsize=8.5, fontweight='bold', color='darkred')

    # 4. Voeg tekstlabels toe boven de top 3 pieken
    for idx in top3_gen:
        plt.annotate(f"{E_vals[idx]:.2f} meV", 
                     (E_vals[idx], sigma_gen[idx]),
                     textcoords="offset points", xytext=(0, 8), ha='center', 
                     fontsize=9, fontweight='bold', color='red')

    for idx in top3_born:
        plt.annotate(f"{E_vals[idx]:.2f} meV", 
                     (E_vals[idx], sigma_born[idx]),
                     textcoords="offset points", xytext=(0, 8), ha='center', 
                     fontsize=9, color='purple')

    # 5. Print de top 3 pieken (gesorteerd van hoog naar laag) in de console
    print("\n--- TOP 3 HOOGSTE PIEKEN (NUMEROV) ---")
    for idx in top3_gen[::-1]:
        print(f"E = {E_vals[idx]:.4f} meV | sigma = {sigma_gen[idx]:.2f} Å²")

    print("\n--- TOP 3 HOOGSTE PIEKEN (BORN) ---")
    for idx in top3_born[::-1]:
        print(f"E = {E_vals[idx]:.4f} meV | sigma = {sigma_born[idx]:.2f} Å²")

    # In je subplot 1 (Verstrooiingsdoorsnede):
    # plt.plot(E_vals, sigma_born, 'm--', linewidth=2.0, label="Born-benadering")
    for l in range(l_max + 1):
        plt.plot(E_vals, partial_gen[l], linestyle='--', alpha=0.7, label=f"l = {l}")
    plt.title("Gegeneraliseerde Numerov: Verstrooiing & Faseverschuivingen", fontweight='bold')
    plt.xlabel("Energie E (meV)")
    plt.ylabel("Doorsnede (Å²)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.show()
    
    # --- FWHM BEREKENING VOOR ALLE L-WAARDEN (0 t/m l_max) ---
    print("\n" + "="*70)
    print(f"{'Impulsmoment (l)':<18} | {'FWHM Breedte Γ (meV)':<20}")
    print("="*70)

    # Dictionary om de breedtes op te slaan als je ze later nog wilt gebruiken
    gamma_results = {}

    for l in range(l_max + 1):
        sig_l = partial_gen[l]
        idx_max = np.argmax(sig_l)
        peak_sigma = sig_l[idx_max]
        half_max = peak_sigma / 2.0
        
        # Linker snijpunt op halve hoogte
        left_side = sig_l[:idx_max]
        E_l_grid = E_vals[:idx_max]
        if len(left_side) > 0:
            idx_l_side = np.argmin(np.abs(left_side - half_max))
            E1 = E_l_grid[idx_l_side]
        else:
            E1 = E_vals[idx_max]
            
        # Rechter snijpunt op halve hoogte
        right_side = sig_l[idx_max:]
        E_r_grid = E_vals[idx_max:]
        if len(right_side) > 0:
            idx_r_side = np.argmin(np.abs(right_side - half_max))
            E2 = E_r_grid[idx_r_side]
        else:
            E2 = E_vals[idx_max]
            
        # Bereken de breedte Gamma
        gamma = E2 - E1
        gamma_results[l] = gamma
        
        print(f"l = {l:<15} | {gamma:<20.4f}")

    print("="*70 + "\n")

    # --- PLOT 2: KLASSIEK ---
    plt.figure(figsize=(10, 6))
    plt.plot(E_vals, sigma_cls, 'r-', linewidth=2.5, label=r"$\sigma_{\text{tot}}$")
    for l in range(l_max + 1):
        plt.plot(E_vals, partial_cls[l], linestyle='--', alpha=0.7, label=f"l = {l}")
    plt.title("Klassieke Numerov: Verstrooiing & Faseverschuivingen", fontweight='bold')
    plt.xlabel("Energie E (meV)")
    plt.ylabel("Doorsnede (Å²)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.show()


def compute_phase_shifts_and_cross_sections(method_type, E_range, l_max=6, r_min=0.5, r_max=15.0, n_points=3000):
    r_grid = np.linspace(r_min, r_max, n_points)
    
    const_asymp = np.sqrt(FACTOR_2M_HBAR2 * EPSILON / 25.0)
    u0 = np.exp(-const_asymp * (r_grid[0]**(-5)))
    u1 = np.exp(-const_asymp * (r_grid[1]**(-5)))
    
    sigma_tot_list = []
    partial_sigma = np.zeros((l_max + 1, len(E_range)))
    delta_matrix = np.zeros((l_max + 1, len(E_range))) # Opslag voor faseverschuivingen
    
    for idx_E, E in enumerate(E_range):
        k2_val = FACTOR_2M_HBAR2 * E
        sum_cross_section = 0.0
        
        for l in range(l_max + 1):
            F_arr = F_scattering(l, r_grid, E)
            
            if method_type == 'generalized':
                u_sol = generalized_numerov_prof_forward(r_grid, F_arr, u0, u1)
            elif method_type == 'classical':
                k2_arr = -F_arr
                u_sol = classical_numerov_forward(r_grid, k2_arr, u0, u1)
                
            delta_l = calculate_phase_shift(l, E, r_grid, u_sol)
            delta_matrix[l, idx_E] = delta_l  # Sla op in radialen
            
            sigma_l = (4.0 * np.pi / k2_val) * (2.0 * l + 1.0) * (np.sin(delta_l)**2)
            partial_sigma[l, idx_E] = sigma_l * CONV_ANG_SQ
            sum_cross_section += sigma_l
            
        sigma_tot_list.append(sum_cross_section * CONV_ANG_SQ)
        
    return E_range, np.array(sigma_tot_list), delta_matrix

def compute_phase_shifts_for_energies(method_type, target_energies, l_max=6, r_min=0.5, r_max=15.0, n_points=3000):
    r_grid = np.linspace(r_min, r_max, n_points)
    
    const_asymp = np.sqrt(FACTOR_2M_HBAR2 * EPSILON / 25.0)
    u0 = np.exp(-const_asymp * (r_grid[0]**(-5)))
    u1 = np.exp(-const_asymp * (r_grid[1]**(-5)))
    
    l_vals = np.arange(0, l_max + 1)
    delta_results = {}
    
    for E in target_energies:
        delta_list = []
        for l in l_vals:
            F_arr = F_scattering(l, r_grid, E)
            
            if method_type == 'generalized':
                u_sol = generalized_numerov_prof_forward(r_grid, F_arr, u0, u1)
            elif method_type == 'classical':
                k2_arr = -F_arr
                u_sol = classical_numerov_forward(r_grid, k2_arr, u0, u1)
            else:
                raise ValueError("Onbekende methode type")
                
            delta_l = calculate_phase_shift(l, E, r_grid, u_sol)
            delta_list.append(np.degrees(delta_l)) # Omzetten naar graden
            
        delta_results[E] = np.array(delta_list)
        
    return l_vals, delta_results



def compute_born_phase_shifts_with_printing(target_energies, l_max=6):
    """
    Berekent en print de Born faseverschuivingen delta_l in radialen en graden 
    voor opgegeven doel-energieën.
    """
    l_vals = np.arange(0, l_max + 1)
    born_delta_results = {}
    
    print(f"\n==================================================")
    print(f" RESULTATEN FASEVERSCHUIVINGEN: BORN-BENADERING")
    print(f"==================================================")
    print(f"{'E (meV)':<10} | {'l':<5} | {'Delta_l (rad)':<15} | {'Delta_l (deg)':<15}")
    print("-" * 55)
    
    for E in target_energies:
        k = np.sqrt(FACTOR_2M_HBAR2 * E)
        delta_list = []
        
        for l in l_vals:
            def integrand(r):
                V = lennard_jones_potential(r)
                jl = sp.spherical_jn(l, k * r)
                return V * (jl**2) * (r**2)
                
            integral, _ = quad(integrand, 0.5, 20.0, limit=100)
            delta_born_l = -FACTOR_2M_HBAR2 * k * integral
            delta_deg = np.degrees(delta_born_l)
            delta_list.append(delta_deg)
            
            # Print op het scherm
            print(f"{E:<10.3f} | {l:<5} | {delta_born_l:<15.5f} | {delta_deg:<15.2f}")
            
        born_delta_results[E] = np.array(delta_list)
        
    return l_vals, born_delta_results

# =============================================================================
# PLOT VAN L-WAARDE VERSUS FASEVERSCHUIVING (Numerov vs. Born)
# =============================================================================
if __name__ == "__main__":
    target_energies = [0.1, 1.237, 2.374]
    l_max = 6
    
    # 1. Bereken en print Numerov faseverschuivingen
    l_vals, delta_res_gen = compute_phase_shifts_for_energies('generalized', target_energies, l_max=l_max)
    
    # 2. Bereken en print Born faseverschuivingen (met console print)
    l_vals_born, delta_res_born = compute_born_phase_shifts_with_printing(target_energies, l_max=l_max)
    
    # 3. Plot beide methoden in dezelfde figuur ter vergelijking
    plt.figure(figsize=(11, 6))
    
    markers = ['o', 's', '^']
    styles = ['-', '--', '-.']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for idx, E in enumerate(target_energies):
        # Numerov (Exacte/Numerieke referentie) - getrokken/gestreepte lijnen
        plt.plot(l_vals, delta_res_gen[E], marker=markers[idx], linestyle=styles[idx], 
                 color=colors[idx], linewidth=2, markersize=8, label=f"Numerov ($E = {E}$ meV)")
        
        # Born-benadering - stippellijnen met een 'x' marker zodat ze makkelijk te onderscheiden zijn
        plt.plot(l_vals_born, delta_res_born[E], marker='x', linestyle=':', 
                 color=colors[idx], linewidth=1.5, markersize=6, label=f"Born ($E = {E}$ meV)")
        
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.title("Vergelijking: Numerov vs. Born faseverschuiving $\\delta_l$ versus $l$", fontsize=13, fontweight='bold')
    plt.xlabel("Impulsmoment $l$", fontsize=12)
    plt.ylabel("Faseverschuiving $\\delta_l$ (graden)", fontsize=12)
    plt.xticks(l_vals)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()

# =============================================================================
# PLOT VAN L-WAARDE VERSUS FASEVERSCHUIVING (delta_l in graden)
# =============================================================================
if __name__ == "__main__":
    target_energies = [0.1, 1.237, 2.374]
    l_max = 6
    
    # Bereken voor de gegeneraliseerde methode (pas aan naar 'classical' indien gewenst)
    l_vals, delta_res_gen = compute_phase_shifts_for_energies('generalized', target_energies, l_max=l_max)
    
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^']
    styles = ['-', '--', '-.']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for idx, E in enumerate(target_energies):
        plt.plot(l_vals, delta_res_gen[E], marker=markers[idx], linestyle=styles[idx], 
                 color=colors[idx], linewidth=2, markersize=8, label=f"$E = {E}$ meV")
        
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.title("Gegeneraliseerd Numerov: faseverschuiving $\\delta_l$ versus $l$-waarde per Energie", fontsize=13, fontweight='bold')
    plt.xlabel("Impulsmoment $l$", fontsize=12)
    plt.ylabel("Faseverschuiving $\\delta_l$ (graden)", fontsize=12)
    plt.xticks(l_vals)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plt.show()
    


# =============================================================================
# PLOT VAN L-WAARDE VERSUS FASEVERSCHUIVING (GESLOTEN & APART)
# =============================================================================
if __name__ == "__main__":
    target_energies = [0.1, 1.237, 2.374]
    l_max = 6
    
    # 1. Bereken Numerov faseverschuivingen
    l_vals, delta_res_gen = compute_phase_shifts_for_energies('generalized', target_energies, l_max=l_max)
    
    # 2. Bereken Born faseverschuivingen (inclusief nette print in de console)
    l_vals_born, delta_res_born = compute_born_phase_shifts_with_printing(target_energies, l_max=l_max)
    
    markers = ['o', 's', '^']
    styles = ['-', '--', '-.']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    # --- PLOT 1: GEGENERALISEERD NUMEROV ---
    plt.figure(figsize=(10, 5.5))
    for idx, E in enumerate(target_energies):
        plt.plot(l_vals, delta_res_gen[E], marker=markers[idx], linestyle=styles[idx], 
                 color=colors[idx], linewidth=2, markersize=8, label=f"$E = {E}$ meV")
        
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.title("Gegeneraliseerd Numerov: faseverschuiving $\\delta_l$ versus $l$", fontsize=13, fontweight='bold')
    plt.xlabel("Impulsmoment $l$", fontsize=12)
    plt.ylabel("Faseverschuiving $\\delta_l$ (graden)", fontsize=12)
    plt.xticks(l_vals)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # --- PLOT 2: BORN-BENADERING (APART) ---
    plt.figure(figsize=(10, 5.5))
    for idx, E in enumerate(target_energies):
        plt.plot(l_vals_born, delta_res_born[E], marker=markers[idx], linestyle=styles[idx], 
                 color=colors[idx], linewidth=2, markersize=8, label=f"Born ($E = {E}$ meV)")
        
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.title("Born-benadering: faseverschuiving $\\delta_l$ versus $l$ (Vergelijking)", fontsize=13, fontweight='bold')
    plt.xlabel("Impulsmoment $l$", fontsize=12)
    plt.ylabel("Faseverschuiving $\\delta_l$ (graden)", fontsize=12)
    plt.xticks(l_vals_born)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plt.show()
    


import numpy as np
import matplotlib.pyplot as plt

# Gegevens van de top 3 pieken (absolute afwijkingen ten opzichte van literatuur)
peaks = ['Hoogste piek 1', 'Hoogste piek 2', 'Hoogste piek 3']
err_E = [0.0361, 0.0077, 0.0211]       # Absolute energiefout |ΔE| in meV
err_sigma = [62.69, 30.97, 35.85]      # Absolute doorsnedefout |Δσ| in Å²

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(peaks))
width = 0.45

# --- LINKER SUBPLOT: Absolute Energiefout (Effen Oranje) ---
rects1 = ax1.bar(x, err_E, width, color='darkorange', edgecolor='black', linewidth=1.2, label='Gegeneraliseerde Numerov')
ax1.set_title('Absolute energiefout $|\\Delta E|$ (meV)', fontsize=12, fontweight='bold')
ax1.set_ylabel('$|\\Delta E| = |E_{\\mathrm{gen}} - E_{\\mathrm{lit}}|$ (meV)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(peaks, fontsize=10, fontweight='bold')
ax1.set_ylim(0, max(err_E) * 1.25)
ax1.grid(True, axis='y', linestyle='--', alpha=0.35)

# Getallen boven de staven zetten (Linker plot)
for rect in rects1:
    val = rect.get_height()
    ax1.annotate(f"{val:.4f}",
                 xy=(rect.get_x() + rect.get_width() / 2, val),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='darkred')

# --- RECHTER SUBPLOT: Absolute Doorsnedefout (Oranje met Arcering) ---
rects2 = ax2.bar(x, err_sigma, width, color='darkorange', edgecolor='black', linewidth=1.2, hatch='//', label='Gegeneraliseerde Numerov')
ax2.set_title('Absolute doorsnedefout $|\\Delta \\sigma|$ ($\\mathrm{\\AA}^2$)', fontsize=12, fontweight='bold')
ax2.set_ylabel('$|\\Delta \\sigma| = |\\sigma_{\\mathrm{tot,gen}} - \\sigma_{\\mathrm{lit}}|$ ($\\mathrm{\\AA}^2$)', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(peaks, fontsize=10, fontweight='bold')
ax2.set_ylim(0, max(err_sigma) * 1.25)
ax2.grid(True, axis='y', linestyle='--', alpha=0.35)

# Getallen boven de staven zetten (Rechter plot)
for rect in rects2:
    val = rect.get_height()
    ax2.annotate(f"{val:.2f}",
                 xy=(rect.get_x() + rect.get_width() / 2, val),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='darkred')

# --- Algemene titel & layout ---
plt.suptitle('Afwijkingen van de Gegeneraliseerde Numerov-methode t.o.v. Literatuur (Thijssen)', fontsize=13.5, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# How is the particle trapped within the cetrifugal barriere, visualise.



import numpy as np
import matplotlib.pyplot as plt

# Fysische constanten en parameters voor H-Kr
EPSILON = 5.9          # meV
SIGMA = 1.0            # Eenheid van lengte = 1 sigma (3.75 Å)
FACTOR_2M_HBAR2 = 6.12 # meV^-1 * sigma^-2

def lennard_jones_potential(r):
    """Berekent de pure Lennard-Jones potentiaal V_LJ(r)"""
    r_safe = np.maximum(r, 1e-8)
    term = SIGMA / r_safe
    return EPSILON * (term**12 - 2.0 * term**6)

def effective_potential(l, r):
    """
    Berekent de totale effectiva potentiaal V_eff(r) = V_LJ(r) + V_centrifugaal
    waarbij V_centrifugaal = l(l+1) / (2m/hbar^2 * r^2)
    """
    V_lj = lennard_jones_potential(r)
    # Centrifugale barrière term
    V_cent = (l * (l + 1.0)) / (FACTOR_2M_HBAR2 * r**2)
    return V_lj + V_cent

if __name__ == "__main__":
    # Rooster van r-waarden (start net boven 0 tot 4.5 sigma om de put en barrière goed te zien)
    r_grid = np.linspace(0.8, 4.5, 1000)
    plt.figure(figsize=(10, 6))

    # Koppel energieniveaus aan de overeenkomstige l-waarden (resonanties)
    energy_levels = {
        4: 0.4639,
        5: 1.4077,
        6: 2.5789,
    }
    resonance_colors = {}

    # Plot de effectieve potentiaal voor l = 0 tot l = 6
    for l in range(7):
        V_eff = effective_potential(l, r_grid)
        # Labels en opmaak (voor l = 4, 5, 6 trekken we ze wat dikker omdat dat je resonanties zijn!)
        linewidth = 2.5 if l in [4, 5, 6] else 1.2
        alpha_val = 1.0 if l in [4, 5, 6] else 0.6
        label_text = f"$l = {l}$" + (" (Resonantie)" if l in [4, 5, 6] else "")
        line = plt.plot(
            r_grid, V_eff, linewidth=linewidth, alpha=alpha_val, label=label_text
        )[0]

        if l in energy_levels:
            resonance_colors[l] = line.get_color()

    # Plot de energieniveaus in de bijbehorende kleur van de resonantie en neem ze op in de legenda
    for l, E_res in energy_levels.items():
        plt.plot(
            r_grid,
            np.full_like(r_grid, E_res),
            color=resonance_colors[l],
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=f"$E_{l} = {E_res:.4f}$ meV",
        )

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title(
        r"Effectieve Potentiaal $V_{\text{eff}}(r)$ met Centrifugale Barrière ($l = 0$ t/m $6$)"
    )
    plt.xlabel(r"Radiale afstand $r$ ($\sigma$)", fontsize=11)
    plt.ylabel(r"Potentiaal $V_{\text{eff}}(r)$ (meV)", fontsize=11)
    plt.xlim(0.8, 1.8)
    plt.ylim(-6.5, 15.05)  # Beperk de y-as zodat de put en barrières mooi zichtbaar zijn
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Fysische constanten en parameters voor H-Kr (consistente notatie)
EPSILON = 5.9          # meV
SIGMA = 1.0            # Eenheid van lengte = 1 sigma (3.75 Å)
FACTOR_2M_HBAR2 = 6.12 # meV^-1 * sigma^-2

def lennard_jones_potential(r):
    """Berekent de Lennard-Jones potentiaal volgens de consistente code-notatie"""
    r_safe = np.maximum(r, 1e-8)
    term = SIGMA / r_safe
    return EPSILON * (term**12 - 2.0 * term**6)

def effective_potential(l, r):
    """Berekent de totale effectieve potentiaal V_eff(r) = V_LJ(r) + V_centrifugaal"""
    V_lj = lennard_jones_potential(r)
    V_cent = (l * (l + 1.0)) / (FACTOR_2M_HBAR2 * r**2)
    return V_lj + V_cent

if __name__ == "__main__":
    r_grid = np.linspace(0.8, 4.5, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Koppel energieniveaus aan de overeenkomstige l-waarden (resonanties)[cite: 3]
    energy_levels = {
        4: 0.4639,
        5: 1.4077,
        6: 2.5789,
    }
    resonance_colors = {}

    # Plot de effectieve potentiaal voor l = 0 tot l = 6[cite: 3]
    for l in range(7):
        V_eff = effective_potential(l, r_grid)
        linewidth = 2.5 if l in [4, 5, 6] else 1.2
        alpha_val = 1.0 if l in [4, 5, 6] else 0.6
        label_text = f"$l = {l}$" + (" (Resonantie)" if l in [4, 5, 6] else "")
        line = ax.plot(
            r_grid, V_eff, linewidth=linewidth, alpha=alpha_val, label=label_text
        )[0]

        if l in energy_levels:
            resonance_colors[l] = line.get_color()

    # Plot de energieniveaus als horizontale lijnen[cite: 3]
    for l, E_res in energy_levels.items():
        ax.plot(
            r_grid,
            np.full_like(r_grid, E_res),
            color=resonance_colors[l],
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label = rf"$E_{{l={l}}} = {E_res:.4f}$ meV",
        )

    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title(
        r"Effectieve Potentiaal $V_{\text{eff}}(r)$ met Centrifugale Barrière & Inzoom"
    )
    ax.set_xlabel(r"Radiale afstand $r$ ($\sigma$)", fontsize=11)
    ax.set_ylabel(r"Potentiaal $V_{\text{eff}}(r)$ (meV)", fontsize=11)
    ax.set_xlim(0.8, 4.5)
    ax.set_ylim(-6.5, 15.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)

    # --- INSET PLOT (INZOOM OP DE BARRIÈRES) ---
    # Maak een ingezoomd venster binnen de hoofdplot [x, y, breedte, hoogte]
    axins = ax.inset_axes([0.42, 0.5, 0.5, 0.45])
    
    for l in range(4, 7):
        V_eff = effective_potential(l, r_grid)
        axins.plot(r_grid, V_eff, color=resonance_colors[l], linewidth=2)
        
    for l, E_res in energy_levels.items():
        axins.plot(
            r_grid, 
            np.full_like(r_grid, E_res), 
            color=resonance_colors[l], 
            linestyle="--", 
            linewidth=1.5
        )

    # Stel het zoom-gebied in op de barrières van l = 4, 5, 6
    axins.set_xlim(1.0, 2.2)
    axins.set_ylim(-1.0, 3.5)
    axins.grid(True, alpha=0.3)
    axins.set_title("Inzoom op Barrières", fontsize=9, fontweight='bold')
    
    # Teken verbindingslijnen tussen de hoofdplot en de inset
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.7)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Fysische constanten en parameters voor H-Kr (consistente notatie)
EPSILON = 5.9          # meV
SIGMA = 1.0            # Eenheid van lengte = 1 sigma (3.75 Å)
FACTOR_2M_HBAR2 = 6.12 # meV^-1 * sigma^-2

def lennard_jones_potential(r):
    """Berekent de Lennard-Jones potentiaal volgens de consistente code-notatie"""
    r_safe = np.maximum(r, 1e-8)
    term = SIGMA / r_safe
    return EPSILON * (term**12 - 2.0 * term**6)

def F_scattering(l, r, E):
    """Berekent de functie F(r) voor de Numerov-methode"""
    V_lj = lennard_jones_potential(r)
    V_cent = (l * (l + 1.0)) / (FACTOR_2M_HBAR2 * r**2)
    return FACTOR_2M_HBAR2 * (V_lj + V_cent - E)

def generalized_numerov_prof_forward(r, F_array, u0, u1):
    """Numerov forward integratie voor de radiale golffunctie u(r)"""
    n = len(r)
    h = r[1] - r[0]
    u = np.zeros(n)
    w = np.zeros(n)
    u[0], u[1] = u0, u1
    w[0] = (1.0 - (h**2 / 12.0) * F_array[0]) * u[0]
    w[1] = (1.0 - (h**2 / 12.0) * F_array[1]) * u[1]
    for i in range(1, n - 1):
        w[i+1] = 2.0 * w[i] - w[i-1] + (h**2) * F_array[i] * u[i]
        u[i+1] = w[i+1] / (1.0 - (h**2 / 12.0) * F_array[i+1])
    return u

if __name__ == "__main__":
    r_grid = np.linspace(0.5, 10.0, 3000)
    const_asymp = np.sqrt(FACTOR_2M_HBAR2 * EPSILON / 25.0)
    u0 = np.exp(-const_asymp * (r_grid[0]**(-5)))
    u1 = np.exp(-const_asymp * (r_grid[1]**(-5)))
    
    # Definieer de 3 resonanties en 3 niet-resonante energieën per l-waarde
    comparisons = [
        {"l": 4, "E_res": 0.4639, "E_off": 1.2000, "label": "l = 4"},
        {"l": 5, "E_res": 1.4077, "E_off": 2.2000, "label": "l = 5"},
        {"l": 6, "E_res": 2.5789, "E_off": 3.5000, "label": "l = 6"}
    ]
    
    plt.figure(figsize=(11, 6.5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for idx, comp in enumerate(comparisons):
        l = comp["l"]
        col = colors[idx]
        
        # 1. Resonantie berekening (Vaste lijn)
        F_res = F_scattering(l, r_grid, comp["E_res"])
        u_res = generalized_numerov_prof_forward(r_grid, F_res, u0, u1)
        u_res /= np.max(np.abs(u_res))  # Normalisatie
        
        plt.plot(r_grid, u_res, color=col, linestyle='-', linewidth=2.2,
                 label=f"Resonantie {comp['label']} ($E = {comp['E_res']:.4f}$ meV)")
        
        # 2. Niet-resonantie berekening (Onderbroken lijn / gestreept)
        F_off = F_scattering(l, r_grid, comp["E_off"])
        u_off = generalized_numerov_prof_forward(r_grid, F_off, u0, u1)
        u_off /= np.max(np.abs(u_off))  # Normalisatie
        
        plt.plot(r_grid, u_off, color=col, linestyle='--', linewidth=1.5, alpha=0.75,
                 label=f"Niet-resonant {comp['label']} ($E = {comp['E_off']:.4f}$ meV)")
        
    plt.title("Vergelijking Radiale Golffuncties: Resonantie (vast) vs. Niet-resonant (gestreept)", fontsize=12, fontweight='bold')
    plt.xlabel("Radiale afstand $r$ ($\sigma$)", fontsize=11)
    plt.ylabel("Genormaliseerde amplitude $u(r)$", fontsize=11)
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.xlim(0.5, 6.05)  # Zoom iets in op het interessante gebied waar de put zit
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()

# Here is more 17:30 pieter here
# We take this as back-up in case soemthign goes wrong

# Name: Monsieurs Pieter
# E-mail: @s0224946@ad.ua.ac.be
# Due date: DUE DATE HERE
# People: individual, with ChatGPT assistance.
# Title: test programma voor gevorderde kwantum taak over verstrooiing
# Last edit: 8/11/2026
# Status: Started.


print("hello world")

import numpy as np
import matplotlib.pyplot as plt

# Fysische constanten en parameters voor H-Kr (consistente notatie)
EPSILON = 5.9          # meV
SIGMA = 1.0            # Eenheid van lengte = 1 sigma (3.75 Å)
FACTOR_2M_HBAR2 = 6.12 # meV^-1 * sigma^-2

def lennard_jones_potential(r):
    """Berekent de Lennard-Jones potentiaal volgens de consistente code-notatie"""
    r_safe = np.maximum(r, 1e-8)
    term = SIGMA / r_safe
    return EPSILON * (term**12 - 2.0 * term**6)

def effective_potential(l, r):
    """Berekent de totale effectieve potentiaal V_eff(r) = V_LJ(r) + V_centrifugaal"""
    V_lj = lennard_jones_potential(r)
    V_cent = (l * (l + 1.0)) / (FACTOR_2M_HBAR2 * r**2)
    return V_lj + V_cent

if __name__ == "__main__":
    r_grid = np.linspace(0.8, 4.5, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Koppel energieniveaus aan de overeenkomstige l-waarden (resonanties)[cite: 3]
    energy_levels = {
        4: 0.4639,
        5: 1.4077,
        6: 2.5789,
    }
    resonance_colors = {}

    # Plot de effectieve potentiaal voor l = 0 tot l = 6[cite: 3]
    for l in range(7):
        V_eff = effective_potential(l, r_grid)
        linewidth = 2.5 if l in [4, 5, 6] else 1.2
        alpha_val = 1.0 if l in [4, 5, 6] else 0.6
        label_text = f"$l = {l}$" + (" (Resonantie)" if l in [4, 5, 6] else "")
        line = ax.plot(
            r_grid, V_eff, linewidth=linewidth, alpha=alpha_val, label=label_text
        )[0]

        if l in energy_levels:
            resonance_colors[l] = line.get_color()

    # Plot de energieniveaus als horizontale lijnen[cite: 3]
    for l, E_res in energy_levels.items():
        ax.plot(
            r_grid,
            np.full_like(r_grid, E_res),
            color=resonance_colors[l],
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label = rf"$E_{{l={l}}} = {E_res:.4f}$ meV",
        )

    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title(
        r"Effectieve Potentiaal $V_{\text{eff}}(r)$ met Centrifugale Barrière & Inzoom"
    )
    ax.set_xlabel(r"Radiale afstand $r$ ($\sigma$)", fontsize=11)
    ax.set_ylabel(r"Potentiaal $V_{\text{eff}}(r)$ (meV)", fontsize=11)
    ax.set_xlim(0.8, 4.5)
    ax.set_ylim(-6.5, 15.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)

    # --- INSET PLOT (INZOOM OP DE BARRIÈRES) ---
    # Maak een ingezoomd venster binnen de hoofdplot [x, y, breedte, hoogte]
    axins = ax.inset_axes([0.42, 0.5, 0.5, 0.45])
    
    for l in range(4, 7):
        V_eff = effective_potential(l, r_grid)
        axins.plot(r_grid, V_eff, color=resonance_colors[l], linewidth=2)
        
    for l, E_res in energy_levels.items():
        axins.plot(
            r_grid, 
            np.full_like(r_grid, E_res), 
            color=resonance_colors[l], 
            linestyle="--", 
            linewidth=1.5
        )

    # Stel het zoom-gebied in op de barrières van l = 4, 5, 6
    axins.set_xlim(1.0, 2.2)
    axins.set_ylim(-1.0, 3.5)
    axins.grid(True, alpha=0.3)
    axins.set_title("Inzoom op Barrières", fontsize=9, fontweight='bold')
    
    # Teken verbindingslijnen tussen de hoofdplot en de inset
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.7)

    plt.tight_layout()
    plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Fysische constanten en parameters voor H-Kr (consistente notatie)
EPSILON = 5.9          # meV
SIGMA = 1.0            # Eenheid van lengte = 1 sigma (3.75 Å)
FACTOR_2M_HBAR2 = 6.12 # meV^-1 * sigma^-2

def lennard_jones_potential(r):
    """Berekent de Lennard-Jones potentiaal volgens de consistente code-notatie"""
    r_safe = np.maximum(r, 1e-8)
    term = SIGMA / r_safe
    return EPSILON * (term**12 - 2.0 * term**6)

def F_scattering(l, r, E):
    """Berekent de functie F(r) voor de Numerov-methode"""
    V_lj = lennard_jones_potential(r)
    V_cent = (l * (l + 1.0)) / (FACTOR_2M_HBAR2 * r**2)
    return FACTOR_2M_HBAR2 * (V_lj + V_cent - E)

def generalized_numerov_prof_forward(r, F_array, u0, u1):
    """Numerov forward integratie voor de radiale golffunctie u(r)"""
    n = len(r)
    h = r[1] - r[0]
    u = np.zeros(n)
    w = np.zeros(n)
    u[0], u[1] = u0, u1
    w[0] = (1.0 - (h**2 / 12.0) * F_array[0]) * u[0]
    w[1] = (1.0 - (h**2 / 12.0) * F_array[1]) * u[1]
    for i in range(1, n - 1):
        w[i+1] = 2.0 * w[i] - w[i-1] + (h**2) * F_array[i] * u[i]
        u[i+1] = w[i+1] / (1.0 - (h**2 / 12.0) * F_array[i+1])
    return u

if __name__ == "__main__":
    r_grid = np.linspace(0.5, 10.0, 3000)
    const_asymp = np.sqrt(FACTOR_2M_HBAR2 * EPSILON / 25.0)
    u0 = np.exp(-const_asymp * (r_grid[0]**(-5)))
    u1 = np.exp(-const_asymp * (r_grid[1]**(-5)))
    
    # Definieer de 3 resonanties en 3 niet-resonante energieën per l-waarde
    comparisons = [
        {"l": 4, "E_res": 0.4639, "E_off": 1.2000, "label": "l = 4"},
        {"l": 5, "E_res": 1.4077, "E_off": 2.2000, "label": "l = 5"},
        {"l": 6, "E_res": 2.5789, "E_off": 3.5000, "label": "l = 6"}
    ]
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    # ==========================================
    # FIGUUR 1: Resonantie-energieën
    # ==========================================
    plt.figure(figsize=(10, 5.5))
    for idx, comp in enumerate(comparisons):
        l = comp["l"]
        col = colors[idx]
        
        F_res = F_scattering(l, r_grid, comp["E_res"])
        u_res = generalized_numerov_prof_forward(r_grid, F_res, u0, u1)
        u_res /= np.max(np.abs(u_res))  # Normalisatie
        
        plt.plot(r_grid, u_res, color=col, linestyle='-', linewidth=2.2,
                 label=f"Resonantie {comp['label']} ($E = {comp['E_res']:.4f}$ meV)")
        
    plt.title("Radiale golffuncties bij resonantie-energieën", fontsize=12, fontweight='bold')
    plt.xlabel("Radiale afstand $r$ ($\sigma$)", fontsize=11)
    plt.ylabel("Genormaliseerde amplitude $u(r)$", fontsize=11)
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # FIGUUR 2: Niet-resonante energieën
    # ==========================================
    plt.figure(figsize=(10, 5.5))
    for idx, comp in enumerate(comparisons):
        l = comp["l"]
        col = colors[idx]
        
        F_off = F_scattering(l, r_grid, comp["E_off"])
        u_off = generalized_numerov_prof_forward(r_grid, F_off, u0, u1)
        u_off /= np.max(np.abs(u_off))  # Normalisatie
        
        plt.plot(r_grid, u_off, color=col, linestyle='--', linewidth=2.0,
                 label=f"Niet-resonant ($E = {comp['E_off']:.4f}$ meV)")
        
    plt.title("Radiale golffuncties bij niet-resonante energieën", fontsize=12, fontweight='bold')
    plt.xlabel("Radiale afstand $r$ ($\sigma$)", fontsize=11)
    plt.ylabel("Genormaliseerde amplitude $u(r)$", fontsize=11)
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()
