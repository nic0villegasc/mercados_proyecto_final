import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import calendar

# --- Funciones Auxiliares para cada tipo de Gráfico de Embalses ---

def _plot_embalses_anual(df_periodo, fecha, ax):
    """
    Genera el gráfico de la evolución mensual del nivel de embalses para un año específico.
    """
    # Usamos pivot con el mes del año (1-12) como índice.
    df_pivot = df_periodo.pivot(index='mes_del_año', columns='generador', values='nivel_embalse_gwh')

    df_pivot.plot(ax=ax, marker='o', linestyle='-')

    ax.set_title(f'Evolución Mensual del Nivel de Embalses - Año {fecha}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Mes del Año', fontsize=12)
    
    ax.set_xticks(df_pivot.index)
    # Se genera la etiqueta solo si el mes 'i' está en el rango válido.
    nombres_meses = [calendar.month_abbr[i] for i in df_pivot.index if 0 < i < 13]
    ax.set_xticklabels(nombres_meses)


def _plot_embalses_rango_anual(df_periodo, fecha, ax):
    """
    Genera un resumen del nivel de embalse a fin de año (mes 12) para un rango de años.
    """
    df_fin_de_anio = df_periodo[df_periodo['mes_del_año'] == 12].copy()

    if df_fin_de_anio.empty:
        ax.text(0.5, 0.5, 'No hay datos del mes 12 para este período.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        print("Advertencia: No se encontraron datos para el mes 12 en el período seleccionado.")
        return

    df_pivot = df_fin_de_anio.pivot(index='año', columns='generador', values='nivel_embalse_gwh')

    df_pivot.plot(ax=ax, marker='o', linestyle='-')
    
    ax.set_title(f'Nivel de Embalse a Fin de Año - Período {fecha}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Año', fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# --- Función Principal ---

def generar_grafico_embalses(periodo, ruta_csv, carpeta_salida):
    """
    Genera un gráfico de nivel de embalses a partir de un índice de mes contínuo.
    - "YYYY": Evolución mensual para ese año.
    - "YYYY-YYYY": Evolución del nivel a fin de año para ese rango.
    """
    try:
        df = pd.read_csv(ruta_csv, sep=';')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en: {ruta_csv}")
        return

    required_cols = {'generador', 'mes', 'nivel_embalse_mwh'}
    if not required_cols.issubset(df.columns):
        print(f"Error: El archivo CSV debe contener las columnas: {', '.join(required_cols)}")
        return
    
    # --- PASO CLAVE: Traducir el mes contínuo (1-120) a año y mes del año (1-12) ---
    start_year = 2025
    # El mes 1 a 12  -> offset 0 -> año 2025. (1-1)//12 = 0
    # El mes 13 a 24 -> offset 1 -> año 2026. (13-1)//12 = 1
    df['año'] = start_year + (df['mes'] - 1) // 12
    # El mes 1 -> (1-1)%12+1 = 1. El mes 13 -> (13-1)%12+1 = 1.
    df['mes_del_año'] = (df['mes'] - 1) % 12 + 1
    
    df['nivel_embalse_gwh'] = df['nivel_embalse_mwh'] / 1000

    try:
        if len(periodo) == 4 and periodo.isdigit(): # Año (YYYY)
            df_periodo = df[df['año'] == int(periodo)].copy()
            plot_func = _plot_embalses_anual
        elif len(periodo) == 9 and periodo.count('-') == 1: # Rango de Años (YYYY-YYYY)
            start_year_filter, end_year_filter = [int(p) for p in periodo.split('-')]
            mask = (df['año'] >= start_year_filter) & (df['año'] <= end_year_filter)
            df_periodo = df[mask].copy()
            plot_func = _plot_embalses_rango_anual
        else:
            raise ValueError(f"Formato de período '{periodo}' no es válido para embalses. Use 'YYYY' o 'YYYY-YYYY'.")
    except Exception as e:
        print(f"Error: {e}"); return

    if df_periodo.empty:
        print(f"No se encontraron datos de embalses para el período: {periodo}"); return

    fig, ax = plt.subplots(figsize=(16, 8))
    
    plot_func(df_periodo, periodo, ax)
    
    ax.set_ylabel('Nivel del Embalse (GWh)', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Embalse')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        
    ruta_guardado = os.path.join(carpeta_salida, f'grafico_nivel_embalses_{periodo}.png')
    # plt.savefig(ruta_guardado, dpi=300)
    print(f"Gráfico guardado exitosamente en: {ruta_guardado}")
    
    plt.show()

# --- CONFIGURACIÓN Y EJECUCIÓN ---
if __name__ == "__main__":
    # --- ¡Elige el período que quieres analizar! ---
    # PERIODO_SELECCIONADO = "2025"      # Para ver la evolución mensual de un año
    PERIODO_SELECCIONADO = "2025-2034"  # Para ver la tendencia a largo plazo (nivel a fin de año)
    
    RUTA_EMBALSES_CSV = os.path.join("resultados_simulacion", "resultados_nivel_embalse_mensual.csv")
    CARPETA_GRAFICOS = os.path.join("resultados_simulacion", "graficos")

    generar_grafico_embalses(PERIODO_SELECCIONADO, RUTA_EMBALSES_CSV, CARPETA_GRAFICOS)