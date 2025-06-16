import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Leer archivo Excel
archivo = 'Péndulo R7.xlsx'
df = pd.read_excel(archivo, sheet_name='Hoja1')

# Renombrar y limpiar columnas
df.columns = ['t', 'x', 'y']
df = df.dropna()
df['t'] = pd.to_numeric(df['t'], errors='coerce')
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df = df.dropna(subset=['t', 'x'])

# Extraer datos
tiempos = df['t'].values
posiciones = df['x'].values

# Modelo
def modelo(t, A, w, phi, C):
    return A * np.cos(w * t + phi) + C

# Estimaciones iniciales
A0 = (max(posiciones) - min(posiciones)) / 2
C0 = np.mean(posiciones)
T0 = tiempos[np.argmax(posiciones)] - tiempos[np.argmin(posiciones)]
w0 = 2 * np.pi / T0 if T0 != 0 else 2 * np.pi
phi0 = 0

# Ajuste
param_opt, _ = curve_fit(modelo, tiempos, posiciones, p0=[A0, w0, phi0, C0])
A, w, phi, C = param_opt

# Mostrar resultados
print(f"Amplitud A      = {A:.5f} m")
print(f"Frecuencia w    = {w:.5f} rad/s")
print(f"Fase inicial φ  = {phi:.5f} rad")
print(f"Desplazamiento C= {C:.5f} m")
print(f"Periodo T       = {2 * np.pi / w:.5f} s")

# Graficar
t_fit = np.linspace(min(tiempos), max(tiempos), 500)
x_fit = modelo(t_fit, A, w, phi, C)

plt.plot(tiempos, posiciones, 'o', label='Datos experimentales')
plt.plot(t_fit, x_fit, '-', label='Ajuste')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (px)')
plt.legend()
plt.grid(True)
plt.title('Ajuste por mínimos cuadrados al MAS')
plt.show()