import numpy as np
import matplotlib.pyplot as plt
import math
import os
from typing import List, Tuple

# Clase principal para el análisis estadístico
class AnalisisEstadistico:
    def __init__(self, datos: List[float]):
        # Se ordenan los datos y se inicializan los atributos principales
        self.datos = sorted(datos)
        self.n = len(datos)
        self.min = min(datos)
        self.max = max(datos)
        self.amplitud_total = self.max - self.min
        # Se calcula el número de clases usando la fórmula de Sturges
        self.k = math.ceil(1 + 3.322 * math.log10(self.n))
        # Tamaño del intervalo entre clases
        self.intervalo = math.ceil(self.amplitud_total / self.k)
        self.intervals = self._generar_intervalos()
        self.f_abs = self._frecuencia_absoluta()
        self.f_acum = np.cumsum(self.f_abs).tolist()
        self.f_rel = [round(f / self.n, 4) for f in self.f_abs]
        self.f_rel_acum = np.cumsum(self.f_rel).tolist()

    # Genera los intervalos de clase como tuplas (inicio, fin)
    def _generar_intervalos(self) -> List[Tuple[int, int]]:
        intervalos = []
        inicio = self.min
        for _ in range(self.k):
            fin = inicio + self.intervalo - 1
            intervalos.append((inicio, fin))
            inicio = fin + 1
        return intervalos

    # Cuenta cuántos datos caen en cada intervalo
    def _frecuencia_absoluta(self) -> List[int]:
        return [sum(inf <= x <= sup for x in self.datos) for inf, sup in self.intervals]

    # Calcula los puntos medios de cada intervalo
    def punto_medio(self) -> List[float]:
        return [(a + b) / 2 for a, b in self.intervals]

    # Calcula la media ponderada para datos agrupados
    def media(self) -> float:
        pm = self.punto_medio()
        return round(sum(f * x for f, x in zip(self.f_abs, pm)) / self.n, 2)

    # Calcula la mediana estimada a partir de las frecuencias acumuladas
    def mediana(self) -> float:
        mitad = self.n / 2
        for i, f_ac in enumerate(self.f_acum):
            if f_ac >= mitad:
                L = self.intervals[i][0] - 0.5
                F_antes = self.f_acum[i - 1] if i > 0 else 0
                f = self.f_abs[i]
                return round(L + ((mitad - F_antes) / f) * self.intervalo, 2)
        return None

    # Calcula la moda para datos agrupados con fórmula de interpolación
    def moda(self) -> float:
        i = self.f_abs.index(max(self.f_abs))
        L = self.intervals[i][0] - 0.5
        f1 = self.f_abs[i]
        f0 = self.f_abs[i - 1] if i > 0 else 0
        f2 = self.f_abs[i + 1] if i < self.k - 1 else 0
        return round(L + ((f1 - f0) / ((f1 - f0) + (f1 - f2))) * self.intervalo, 2)

    # Varianza ponderada usando puntos medios
    def varianza(self) -> float:
        pm = self.punto_medio()
        m = self.media()
        return round(sum(f * ((x - m) ** 2) for f, x in zip(self.f_abs, pm)) / self.n, 2)

    # Desviación estándar como raíz cuadrada de la varianza
    def desviacion_estandar(self) -> float:
        return round(math.sqrt(self.varianza()), 2)

    # Coeficiente de variación = desviación / media
    def coef_variacion(self) -> float:
        return round(self.desviacion_estandar() / self.media(), 4)

    # Cuartil usando interpolación sobre los datos ordenados
    def cuartil(self, q: int) -> float:
        pos = (q * self.n) / 4
        return self.datos[int(pos) - 1] if pos.is_integer() else round(np.interp(pos, range(1, self.n + 1), self.datos), 2)

    # Percentil por interpolación
    def percentil(self, p: int) -> float:
        pos = (p * self.n) / 100
        return self.datos[int(pos) - 1] if pos.is_integer() else round(np.interp(pos, range(1, self.n + 1), self.datos), 2)

    # Asimetría de Pearson (3 veces la diferencia entre media y mediana dividida por la desviación)
    def asimetria_pearson(self) -> float:
        media = self.media()
        mediana = self.mediana()
        sigma = self.desviacion_estandar()
        return round(3 * (media - mediana) / sigma, 2)

    # Curtosis (exceso), mide si los datos son más o menos concentrados que la normal
    def curtosis(self) -> float:
        media = np.mean(self.datos)
        sigma = np.std(self.datos)
        if sigma == 0:
            return 0
        g2 = np.mean([(x - media) ** 4 for x in self.datos]) / (sigma ** 4) - 3
        return round(g2, 2)

    # Resumen general de todos los estadísticos en un diccionario
    def resumen(self) -> dict:
        return {
            "Media": self.media(),
            "Mediana": self.mediana(),
            "Moda": self.moda(),
            "Rango": self.amplitud_total,
            "Varianza": self.varianza(),
            "Desviación estándar": self.desviacion_estandar(),
            "Coef. de variación": self.coef_variacion(),
            "Q1": self.cuartil(1),
            "Q2 (Mediana)": self.cuartil(2),
            "Q3": self.cuartil(3),
            "P10": self.percentil(10),
            "P90": self.percentil(90),
            "Asimetría de Pearson": self.asimetria_pearson(),
            "Curtosis (exceso)": self.curtosis()
        }

    # Construye la tabla de frecuencias agrupadas
    def tabla_frecuencias(self):
        tabla = []
        for i, (a, b) in enumerate(self.intervals):
            fila = {
                "Intervalo": f"{a}-{b}",
                "f": self.f_abs[i],
                "F": self.f_acum[i],
                "f%": round(self.f_rel[i] * 100, 2),
                "F%": round(self.f_rel_acum[i] * 100, 2)
            }
            tabla.append(fila)
        return tabla

    # Genera y guarda todas las gráficas en la carpeta 'graficas'
    def graficas(self):
        os.makedirs("graficas", exist_ok=True)
        etiquetas = [f"{a}-{b}" for a, b in self.intervals]
        medios = self.punto_medio()

        # Histograma
        plt.bar(etiquetas, self.f_abs, color='skyblue', edgecolor='black')
        plt.title("Histograma")
        plt.xlabel("Intervalos")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig("graficas/histograma.png")
        plt.close()

        # Polígono de frecuencias
        plt.plot(medios, self.f_abs, marker='o', linestyle='-', color='green')
        plt.title("Polígono de Frecuencias")
        plt.xlabel("Punto Medio")
        plt.ylabel("Frecuencia")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("graficas/poligono.png")
        plt.close()

        # Ojiva (frecuencia acumulada)
        limites_sup = [b for _, b in self.intervals]
        plt.plot(limites_sup, self.f_acum, marker='o', linestyle='-', color='red')
        plt.title("Ojiva")
        plt.xlabel("Límite superior")
        plt.ylabel("Frecuencia Acumulada")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("graficas/ojiva.png")
        plt.close()

        # Boxplot (diagrama de caja)
        plt.boxplot(self.datos, vert=False)
        plt.title("Diagrama de Caja (Boxplot)")
        plt.xlabel("Datos")
        plt.tight_layout()
        plt.savefig("graficas/boxplot.png")
        plt.close()

# ================================
# EJECUCIÓN PRINCIPAL
# ================================

if __name__ == "__main__":
    # Datos del ejercicio
    datos = [25, 32, 28, 45, 30, 27, 38, 40, 33, 29, 35, 36, 26, 31, 34, 42, 37, 43, 41, 39,
             24, 30, 29, 31, 33, 30, 35, 28, 30, 32, 36, 38, 27, 26, 25, 34, 40, 39, 31, 30,
             29, 33, 32, 35, 37, 34, 28, 26, 25, 24, 45, 46, 42, 41, 43, 44, 47, 48, 49, 50]

    analisis = AnalisisEstadistico(datos)

    print("\nResumen estadístico completo:")
    for k, v in analisis.resumen().items():
        print(f"{k}: {v}")

    print("\nTabla de frecuencias:")
    for fila in analisis.tabla_frecuencias():
        print(fila)

    # Generar las gráficas en PNG
    analisis.graficas()
    print("\nTodas las gráficas fueron guardadas en la carpeta 'graficas/'.")
