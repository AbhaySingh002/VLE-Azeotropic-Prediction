import numpy as np
from scipy.optimize import fsolve
import pandas as pd

# Antoine constants (log10(P_mmHg) = A - B / (T_C + C))
A_eth, B_eth, C_eth = 8.20417, 1642.89, 230.300  # Ethanol
A_wat, B_wat, C_wat = 8.07131, 1730.63, 233.426  # Water

def Psat_ethanol(Tc):
    return 10 ** (A_eth - B_eth / (Tc + C_eth))

def Psat_water(Tc):
    return 10 ** (A_wat - B_wat / (Tc + C_wat))

# NRTL Parameters (tau_ij = a_ij + b_ij / T_K, alpha=0.3)
a12, b12 = -0.81, 246  # Ethanol(1)-Water(2)
a21, b21 = 3.46, -586
alpha = 0.3

def get_gammas(x1, Tk):
    x2 = 1 - x1
    tau12_ = a12 + b12 / Tk
    tau21_ = a21 + b21 / Tk
    G12 = np.exp(-alpha * tau12_)
    G21 = np.exp(-alpha * tau21_)
    gamma1 = np.exp(x2**2 * (tau21_ * (G21 / (x1 + x2 * G21))**2 + tau12_ * G12 / (x2 + x1 * G12)**2))
    gamma2 = np.exp(x1**2 * (tau12_ * (G12 / (x2 + x1 * G12))**2 + tau21_ * G21 / (x1 + x2 * G21)**2))
    return gamma1, gamma2

def bubble_T(x1, P_mmHg):
    def eq(Tc):
        Tk = Tc + 273.15
        gamma1, gamma2 = get_gammas(x1, Tk)
        return x1 * gamma1 * Psat_ethanol(Tc) + (1 - x1) * gamma2 * Psat_water(Tc) - P_mmHg

    # Improved initial guess: use Antoine equation for pure components or azeotrope
    if x1 == 0:
        T_guess = 100  # Water boiling point
    elif x1 == 1:
        T_guess = 78.3  # Ethanol boiling point
    else:
        # For mixtures, estimate based on pressure and composition
        T_guess = 78 + (100 - 78) * (1 - x1)  # Linear approximation
        # Adjust for pressure: lower P → lower T
        if P_mmHg < 760:
            T_guess -= (760 - P_mmHg) / 760 * 20  # Rough scaling
        elif P_mmHg > 760:
            T_guess += (P_mmHg - 760) / 760 * 10

    try:
        Tc = fsolve(eq, T_guess, xtol=1e-6, maxfev=1000)[0]
        # Check if solution is reasonable (e.g., 50°C < Tc < 150°C)
        if not 50 <= Tc <= 150:
            print(f"Warning: Tc={Tc:.2f}°C for x1={x1:.5f}, P={P_mmHg} mmHg is out of bounds. Using fallback.")
            Tc = T_guess  # Fallback to guess
        return np.round(Tc, 5)
    except Exception as e:
        print(f"fsolve failed for x1={x1:.5f}, P={P_mmHg} mmHg: {e}. Using T_guess.")
        return np.round(T_guess, 5)

# Generate data
P_mmHg_list = [200, 400, 760, 1000, 1500]  # ~0.26 to ~2 atm
x_base = np.round(np.concatenate([np.linspace(0, 0.7, 30), np.linspace(0.7, 0.95, 50), np.linspace(0.95, 1, 20)]), 5)

data = []
for P_mmHg in P_mmHg_list:
    for x1 in x_base:
        if x1 == 0 or x1 == 1:
            Tc = 100 if x1 == 0 else 78.3
            y1 = x1
        else:
            Tc = bubble_T(x1, P_mmHg)
            Tk = Tc + 273.15
            gamma1, _ = get_gammas(x1, Tk)
            y1 = x1 * gamma1 * Psat_ethanol(Tc) / P_mmHg
        P_atm = P_mmHg / 760
        # Round all values to 5 decimal places for final output
        x1 = np.round(x1, 5)
        Tk = np.round(Tk, 5)
        P_atm = np.round(P_atm, 5)
        y1 = np.round(y1, 5)
        data.append([x1, Tk, P_atm, y1])
        print(f"Data point: [x1={x1:.5f}, Tk={Tk:.5f}, P_atm={P_atm:.5f}, y1={y1:.5f}]")  # Debug

df = pd.DataFrame(data, columns=['x1', 'T', 'P', 'y1'])
df = df.astype(float)  # Ensure all numeric
print("DataFrame head:\n", df.head())
print("DataFrame dtypes:\n", df.dtypes)
df.to_csv('Data/vle_data.csv', index=False)
print(f"Generated {len(df)} data points. Saved to Data/vle_data.csv")