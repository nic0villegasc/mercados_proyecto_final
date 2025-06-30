import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import calendar

# --- Las funciones auxiliares (_plot_embalses_anual, _plot_embalses_rango_anual) no necesitan cambios ---

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


# --- Función Principal (Modificada) ---

def generar_grafico_embalses(periodo, ruta_csv, carpeta_salida, escenario):
    """
    Genera un gráfico de nivel de embalses a partir de datos horarios.
    - "YYYY": Evolución mensual para ese año.
    - "YYYY-YYYY": Evolución del nivel a fin de año para ese rango.
    """
    try:
        df = pd.read_csv(ruta_csv, sep=';')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en: {ruta_csv}")
        return

    # --- CAMBIO 1: Validar las nuevas columnas requeridas ---
    required_cols = {'generador', 'fecha_hora', 'nivel_embalse_mwh'}
    if not required_cols.issubset(df.columns):
        print(f"Error: El archivo CSV debe contener las columnas: {', '.join(required_cols)}")
        return
    
    # --- CAMBIO 2: Procesar la columna 'fecha_hora' en lugar de 'mes' ---
    # Convertir la columna de fecha a formato datetime de pandas
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    
    # Para poder graficar, necesitamos agrupar los datos horarios en mensuales.
    # Usaremos el último valor registrado de cada mes como representativo.
    df = df.set_index('fecha_hora')
    # Agrupamos por generador, remuestreamos a frecuencia mensual ('M') y tomamos el último valor.
    df_mensual = df.groupby('generador').resample('M').last().drop(columns='generador').reset_index()

    # Ahora creamos las columnas de 'año' y 'mes_del_año' a partir del índice de fecha
    df_mensual['año'] = df_mensual['fecha_hora'].dt.year
    df_mensual['mes_del_año'] = df_mensual['fecha_hora'].dt.month
    
    df_mensual['nivel_embalse_gwh'] = df_mensual['nivel_embalse_mwh'] / 1000

    # --- El resto de la lógica de filtrado y graficado usa el nuevo df_mensual ---
    try:
        if len(periodo) == 4 and periodo.isdigit(): # Año (YYYY)
            df_periodo = df_mensual[df_mensual['año'] == int(periodo)].copy()
            plot_func = _plot_embalses_anual
        elif len(periodo) == 9 and periodo.count('-') == 1: # Rango de Años (YYYY-YYYY)
            start_year_filter, end_year_filter = [int(p) for p in periodo.split('-')]
            mask = (df_mensual['año'] >= start_year_filter) & (df_mensual['año'] <= end_year_filter)
            df_periodo = df_mensual[mask].copy()
            plot_func = _plot_embalses_rango_anual
        else:
            raise ValueError(f"Formato de período '{periodo}' no es válido para embalses. Use 'YYYY' o 'YYYY-YYYY'.")
    except Exception as e:
        print(f"Error: {e}"); return

    if df_periodo.empty:
        print(f"No se encontraron datos de embalses para el período: {periodo}"); return

    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Añadimos el escenario al título
    fig.suptitle(f'Escenario Hidrológico: {escenario.upper()}', fontsize=20, fontweight='bold')
    
    plot_func(df_periodo, periodo, ax)
    
    ax.set_ylabel('Nivel del Embalse (GWh)', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Embalse')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para el supertítulo

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        
    ruta_guardado = os.path.join(carpeta_salida, f'grafico_nivel_embalses_{periodo}.png')
    plt.savefig(ruta_guardado, dpi=300)
    print(f"Gráfico guardado exitosamente en: {ruta_guardado}")
    
    plt.show()

# --- CONFIGURACIÓN Y EJECUCIÓN ---
if __name__ == "__main__":
    # --- ¡Elige el escenario y período que quieres analizar! ---
    # --- CAMBIO 3: Seleccionar el escenario a graficar ---
    ESCENARIO = "MEDIA"  # Puedes cambiarlo a "SECA" o "HUMEDA"
    
    PERIODO_SELECCIONADO = "2025"      # Para ver la evolución mensual de un año
    # PERIODO_SELECCIONADO = "2025-2034"  # Para ver la tendencia a largo plazo (nivel a fin de año)
    
    # --- CAMBIO 4: Rutas dinámicas basadas en el escenario ---
    CARPETA_BASE = f"resultados_simulacion_{ESCENARIO}"
    RUTA_EMBALSES_CSV = os.path.join(CARPETA_BASE, "resultados_nivel_embalse_horario.csv")
    CARPETA_GRAFICOS = os.path.join(CARPETA_BASE, "graficos")

    generar_grafico_embalses(PERIODO_SELECCIONADO, RUTA_EMBALSES_CSV, CARPETA_GRAFICOS, ESCENARIO)