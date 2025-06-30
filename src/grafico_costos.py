import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import calendar

# --- Las funciones auxiliares (_plot_costos_diarios, etc.) no necesitan cambios ---

def _plot_costos_diarios(df_periodo, fecha, ax):
    """Genera el gráfico de costos marginales horarios para un solo día."""
    df_periodo['hora'] = df_periodo['fecha_hora'].dt.hour
    # Usamos pivot para tener una columna por cada barra, ideal para multi-línea.
    df_pivot = df_periodo.pivot(index='hora', columns='barra', values='costo_marginal_usd_mwh')
    
    df_pivot.plot(ax=ax, marker='.', linestyle='-')
    
    ax.set_title(f'Costo Marginal Horario por Barra - {fecha}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hora del Día', fontsize=12)

def _plot_costos_mensuales(df_periodo, fecha, ax):
    """Genera un resumen del costo marginal promedio diario para un mes."""
    df_periodo['dia'] = df_periodo['fecha_hora'].dt.day
    # Usamos pivot_table para agrupar por día y calcular el promedio (mean).
    df_pivot = df_periodo.pivot_table(index='dia', columns='barra', values='costo_marginal_usd_mwh', aggfunc='mean')
    
    df_pivot.plot(ax=ax, marker='.', linestyle='-')

    mes_nombre = calendar.month_name[int(fecha.split('-')[1])]
    ax.set_title(f'Costo Marginal Promedio Diario por Barra - {mes_nombre} {fecha.split("-")[0]}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Día del Mes', fontsize=12)

def _plot_costos_anuales(df_periodo, fecha, ax):
    """Genera un resumen del costo marginal promedio mensual para un año."""
    df_periodo['mes'] = df_periodo['fecha_hora'].dt.month
    df_pivot = df_periodo.pivot_table(index='mes', columns='barra', values='costo_marginal_usd_mwh', aggfunc='mean')
    
    df_pivot.plot(ax=ax, marker='.', linestyle='-')
    
    ax.set_title(f'Costo Marginal Promedio Mensual por Barra - Año {fecha}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Mes del Año', fontsize=12)
    nombres_meses = [calendar.month_abbr[i] for i in df_pivot.index]
    ax.set_xticklabels(nombres_meses)
    # Asegurarnos de que los ticks coincidan con las etiquetas
    ax.set_xticks(df_pivot.index)


def _plot_costos_rango_anual(df_periodo, fecha, ax):
    """Genera un resumen del costo marginal promedio anual para un rango de años."""
    df_periodo['anio'] = df_periodo['fecha_hora'].dt.year
    df_pivot = df_periodo.pivot_table(index='anio', columns='barra', values='costo_marginal_usd_mwh', aggfunc='mean')

    df_pivot.plot(ax=ax, marker='.', linestyle='-')

    ax.set_title(f'Costo Marginal Promedio Anual por Barra - Período {fecha}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Año', fontsize=12)
    # Forzar a que los ticks del eje X sean enteros para los años
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# --- Función Principal (Modificada) ---

def generar_grafico_costos(periodo, ruta_cmg_csv, carpeta_salida, escenario):
    """
    Genera un gráfico de costos marginales. Soporta los siguientes formatos de período:
    - "YYYY-MM-DD": Gráfico horario.
    - "YYYY-MM": Resumen de promedio diario del mes.
    - "YYYY": Resumen de promedio mensual del año.
    - "YYYY-YYYY": Resumen de promedio anual para el rango de años.
    """
    try:
        df = pd.read_csv(ruta_cmg_csv, sep=';', parse_dates=['fecha_hora'], usecols=['fecha_hora', 'barra', 'costo_marginal_usd_mwh'])
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de costos marginales en: {ruta_cmg_csv}")
        return
    except ValueError:
        print("Error: El CSV de costos no tiene las columnas esperadas. Se necesitan: 'fecha_hora', 'barra', 'costo_marginal_usd_mwh'.")
        return

    try:
        if len(periodo) == 10 and periodo.count('-') == 2: # Día
            df_periodo = df[df['fecha_hora'].dt.date == pd.to_datetime(periodo).date()].copy()
            plot_func = _plot_costos_diarios
        elif len(periodo) == 7 and periodo.count('-') == 1: # Mes
            df_periodo = df[df['fecha_hora'].dt.strftime('%Y-%m') == periodo].copy()
            plot_func = _plot_costos_mensuales
        elif len(periodo) == 4 and periodo.isdigit(): # Año
            df_periodo = df[df['fecha_hora'].dt.year == int(periodo)].copy()
            plot_func = _plot_costos_anuales
        elif len(periodo) == 9 and periodo.count('-') == 1: # Rango de Años
            start_year, end_year = [int(p) for p in periodo.split('-')]
            mask = (df['fecha_hora'].dt.year >= start_year) & (df['fecha_hora'].dt.year <= end_year)
            df_periodo = df[mask].copy()
            plot_func = _plot_costos_rango_anual
        else: raise ValueError()
    except (ValueError, IndexError):
        print(f"Error: Formato de período '{periodo}' no reconocido."); return

    if df_periodo.empty:
        print(f"No se encontraron datos de costos para el período: {periodo}"); return

    # --- CAMBIO 1: Añadir título principal con el nombre del escenario ---
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(f'Escenario Hidrológico: {escenario.upper()}', fontsize=20, fontweight='bold')
    
    plot_func(df_periodo, periodo, ax)
    
    ax.set_ylabel('Costo Marginal (USD/MWh)', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Barra')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Ajustar layout para que el supertítulo no se superponga
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        
    ruta_guardado = os.path.join(carpeta_salida, f'grafico_costo_marginal_{periodo}.png')
    plt.savefig(ruta_guardado, dpi=300)
    print(f"Gráfico guardado exitosamente en: {ruta_guardado}")
    plt.show()

# --- CONFIGURACIÓN Y EJECUCIÓN ---
if __name__ == "__main__":
    # --- ¡Elige el escenario y período que quieres analizar! ---
    # --- CAMBIO 2: Seleccionar el escenario a graficar ---
    ESCENARIO = "MEDIA"  # Puedes cambiarlo a "SECA" o "HUMEDA"

    # Descomenta la línea que quieras usar y comenta las otras.
    # PERIODO_SELECCIONADO = "2025-01-15"   # Vista detallada de un día
    # PERIODO_SELECCIONADO = "2034-07"     # Resumen de un mes (promedio diario)
    PERIODO_SELECCIONADO = "2034"         # Resumen de un año (promedio mensual)
    # PERIODO_SELECCIONADO = "2025-2034"   # Resumen de un rango de años (promedio anual)
    
    # --- CAMBIO 3: Rutas dinámicas basadas en el escenario ---
    CARPETA_BASE = f"resultados_simulacion_{ESCENARIO}"
    RUTA_COSTO_MARGINAL_CSV = os.path.join(CARPETA_BASE, "resultados_costo_marginal.csv")
    CARPETA_GRAFICOS = os.path.join(CARPETA_BASE, "graficos")

    # Pasamos el escenario a la función para usarlo en el título
    generar_grafico_costos(PERIODO_SELECCIONADO, RUTA_COSTO_MARGINAL_CSV, CARPETA_GRAFICOS, ESCENARIO)