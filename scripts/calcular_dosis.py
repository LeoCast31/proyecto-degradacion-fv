import numpy as np
import pandas as pd


def calcular_dosis_degradacion(ruta_csv_entrada, ruta_csv_salida=None):
  """Calcula la dosis de degradación ponderada térmicamente (D_weighted)

  a partir de la iradiación H_POA y la temperatura de celda T_cell.
  """
  print(f"Cargando datos desde: {ruta_csv_entrada}")
  df = pd.read_csv(ruta_csv_entrada)

  # Constantes físicas del proyecto
  k_B = 8.61733262145e-5  # Constante de Boltzmann en eV/K
  E_a = 0.48  # Energía de activación en eV (Defectos Boro-Oxígeno)

  if "H_POA" not in df.columns or "T_cell" not in df.columns:
    raise ValueError(
        "El dataset debe contener las columnas 'H_POA' y 'T_cell'."
    )

  # Conversión de Celsius a Kelvin y cálculo de D
  df["T_cell_K"] = df["T_cell"] + 273.15
  df["D_weighted"] = df["H_POA"] * np.exp(-E_a / (k_B * df["T_cell_K"]))

  print("¡Cálculo de D_weighted completado con éxito!")

  if ruta_csv_salida:
    df.to_csv(ruta_csv_salida, index=False)
    print(f"Archivo guardado en: {ruta_csv_salida}")

  return df


if __name__ == "__main__":
  print("Prueba de ejecución del pipeline de cálculo para la variable D:")
  datos_prueba = pd.DataFrame({
      "H_POA": [500.0, 850.0, 620.0],
      "T_cell": [25.0, 45.0, 35.0],
  })
  datos_prueba.to_csv("datos_ejemplo.csv", index=False)
  df_resultado = calcular_dosis_degradacion(
      "datos_ejemplo.csv", "datos_procesados.csv"
  )
  print(df_resultado[["H_POA", "T_cell", "D_weighted"]])
