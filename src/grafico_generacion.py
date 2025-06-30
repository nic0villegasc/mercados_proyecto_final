import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import calendar

# --- Funciones Auxiliares para cada tipo de Gráfico ---

def _plot_daily(df_gen, df_dem, fecha, ax, colores):
    """Genera el gráfico detallado por hora para un solo día con curva de demanda."""
    df_gen['hora'] = df_gen['fecha_hora'].dt.hour
    df_pivot = df_gen.pivot_table(values='generacion_mw', index='hora', columns='tipo_generador', aggfunc='sum').fillna(0)
    
    columnas_ordenadas = [col for col in ['DIESEL', 'TERMICA', 'HIDRO', 'SOLAR'] if col in df_pivot.columns]
    df_pivot = df_pivot[columnas_ordenadas]
    df_pivot.plot(kind='bar', stacked=True, color=[colores.get(c) for c in df_pivot.columns], ax=ax, width=0.8)

    if df_dem is not None and not df_dem.empty:
        df_dem['hora'] = df_dem['fecha_hora'].dt.hour
        demanda_horaria = df_dem.groupby('hora')['demanda_mw'].sum()
        # PASO CLAVE: Reindexar la demanda para que coincida con el índice de las barras.
        demanda_alineada = demanda_horaria.reindex(df_pivot.index).fillna(0)
        ax.plot(range(len(demanda_alineada)), demanda_alineada.values, color='darkred', linestyle='--', marker='o', label='Demanda')

    ax.set_title(f'Generación vs. Demanda por Hora - {fecha}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hora del Día', fontsize=12)
    ax.set_ylabel('Potencia (MW)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

def _plot_monthly_summary(df_gen, df_dem, fecha, ax, colores):
    """Genera un resumen de generación y demanda diaria para un mes completo."""
    df_gen['dia'] = df_gen['fecha_hora'].dt.day
    df_pivot = df_gen.pivot_table(values='generacion_mw', index='dia', columns='tipo_generador', aggfunc='sum').fillna(0) / 1000 # GWh

    columnas_ordenadas = [col for col in ['DIESEL', 'TERMICA', 'HIDRO', 'SOLAR'] if col in df_pivot.columns]
    df_pivot = df_pivot[columnas_ordenadas]
    df_pivot.plot(kind='bar', stacked=True, color=[colores.get(c) for c in df_pivot.columns], ax=ax, width=0.8)

    if df_dem is not None and not df_dem.empty:
        df_dem['dia'] = df_dem['fecha_hora'].dt.day
        demanda_diaria = df_dem.groupby('dia')['demanda_mw'].sum() / 1000 # GWh
        # PASO CLAVE: Reindexar la demanda para que coincida con el índice de las barras.
        demanda_alineada = demanda_diaria.reindex(df_pivot.index).fillna(0)
        ax.plot(range(len(demanda_alineada)), demanda_alineada.values, color='darkred', linestyle='--', marker='o', label='Demanda')

    mes_nombre = calendar.month_name[int(fecha.split('-')[1])]
    ax.set_title(f'Resumen Diario de Generación vs. Demanda - {mes_nombre} {fecha.split("-")[0]}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Día del Mes', fontsize=12)
    ax.set_ylabel('Energía Total (GWh)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.1f}'))

def _plot_yearly_summary(df_gen, df_dem, fecha, ax, colores):
    """Genera un resumen de generación y demanda mensual para un año completo."""
    df_gen['mes'] = df_gen['fecha_hora'].dt.month
    df_pivot = df_gen.pivot_table(values='generacion_mw', index='mes', columns='tipo_generador', aggfunc='sum').fillna(0) / 1000 # GWh
    
    columnas_ordenadas = [col for col in ['DIESEL', 'TERMICA', 'HIDRO', 'SOLAR'] if col in df_pivot.columns]
    df_pivot = df_pivot[columnas_ordenadas]
    df_pivot.plot(kind='bar', stacked=True, color=[colores.get(c) for c in df_pivot.columns], ax=ax, width=0.8)

    if df_dem is not None and not df_dem.empty:
        df_dem['mes'] = df_dem['fecha_hora'].dt.month
        demanda_mensual = df_dem.groupby('mes')['demanda_mw'].sum() / 1000 # GWh
        # PASO CLAVE: Reindexar la demanda para que coincida con el índice de las barras.
        demanda_alineada = demanda_mensual.reindex(df_pivot.index).fillna(0)
        ax.plot(range(len(demanda_alineada)), demanda_alineada.values, color='darkred', linestyle='--', marker='o', label='Demanda')
    
    ax.set_title(f'Resumen Mensual de Generación vs. Demanda - Año {fecha}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Mes del Año', fontsize=12)
    ax.set_ylabel('Energía Total (GWh)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    nombres_meses = [calendar.month_abbr[i] for i in df_pivot.index]
    ax.set_xticklabels(nombres_meses)

def _plot_yearly_range_summary(df_gen, df_dem, fecha, ax, colores):
    """Genera un resumen de generación y demanda anual para un rango de años."""
    df_gen['anio'] = df_gen['fecha_hora'].dt.year
    df_pivot = df_gen.pivot_table(values='generacion_mw', index='anio', columns='tipo_generador', aggfunc='sum').fillna(0) / 1000000 # TWh
    
    columnas_ordenadas = [col for col in ['DIESEL', 'TERMICA', 'HIDRO', 'SOLAR'] if col in df_pivot.columns]
    df_pivot = df_pivot[columnas_ordenadas]
    df_pivot.plot(kind='bar', stacked=True, color=[colores.get(c) for c in df_pivot.columns], ax=ax, width=0.8)

    if df_dem is not None and not df_dem.empty:
        df_dem['anio'] = df_dem['fecha_hora'].dt.year
        demanda_anual = df_dem.groupby('anio')['demanda_mw'].sum() / 1000000 # TWh
        # PASO CLAVE: Reindexar la demanda para que coincida con el índice de las barras.
        demanda_alineada = demanda_anual.reindex(df_pivot.index).fillna(0)
        ax.plot(range(len(demanda_alineada)), demanda_alineada.values, color='darkred', linestyle='--', marker='o', label='Demanda')

    ax.set_title(f'Resumen Anual de Generación vs. Demanda - Período {fecha}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Año', fontsize=12)
    ax.set_ylabel('Energía Total (TWh)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.2f}'))

# --- Función Principal (SIN CAMBIOS) ---

def generar_grafico_generacion(periodo, ruta_gen_csv, ruta_dem_csv, carpeta_salida):
    try:
        df_gen = pd.read_csv(ruta_gen_csv, sep=';', parse_dates=['fecha_hora'])
        df_gen['tipo_generador'] = df_gen['generador'].apply(lambda x: x.split('-')[0])
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de generación en: {ruta_gen_csv}")
        return

    df_dem = None
    try:
        df_dem = pd.read_csv(ruta_dem_csv, sep=';', parse_dates=['fecha_hora'])
    except FileNotFoundError:
        print(f"Advertencia: No se encontró el archivo de demanda en: {ruta_dem_csv}. Se graficará solo la generación.")

    try:
        df_periodo_gen, df_periodo_dem = None, None
        if len(periodo) == 10 and periodo.count('-') == 2:
            df_periodo_gen = df_gen[df_gen['fecha_hora'].dt.date == pd.to_datetime(periodo).date()].copy()
            if df_dem is not None: df_periodo_dem = df_dem[df_dem['fecha_hora'].dt.date == pd.to_datetime(periodo).date()].copy()
            plot_func = _plot_daily
        elif len(periodo) == 7 and periodo.count('-') == 1:
            df_periodo_gen = df_gen[df_gen['fecha_hora'].dt.strftime('%Y-%m') == periodo].copy()
            if df_dem is not None: df_periodo_dem = df_dem[df_dem['fecha_hora'].dt.strftime('%Y-%m') == periodo].copy()
            plot_func = _plot_monthly_summary
        elif len(periodo) == 4 and periodo.isdigit():
            df_periodo_gen = df_gen[df_gen['fecha_hora'].dt.year == int(periodo)].copy()
            if df_dem is not None: df_periodo_dem = df_dem[df_dem['fecha_hora'].dt.year == int(periodo)].copy()
            plot_func = _plot_yearly_summary
        elif len(periodo) == 9 and periodo.count('-') == 1:
            start_year, end_year = [int(p) for p in periodo.split('-')]
            mask_gen = (df_gen['fecha_hora'].dt.year >= start_year) & (df_gen['fecha_hora'].dt.year <= end_year)
            df_periodo_gen = df_gen[mask_gen].copy()
            if df_dem is not None:
                mask_dem = (df_dem['fecha_hora'].dt.year >= start_year) & (df_dem['fecha_hora'].dt.year <= end_year)
                df_periodo_dem = df_dem[mask_dem].copy()
            plot_func = _plot_yearly_range_summary
        else: raise ValueError()
    except (ValueError, IndexError):
        print(f"Error: Formato de período '{periodo}' no reconocido."); return

    if df_periodo_gen.empty:
        print(f"No se encontraron datos de generación para el período: {periodo}"); return

    colores = {'SOLAR': 'gold', 'TERMICA': 'darkorange', 'HIDRO': 'royalblue', 'DIESEL': 'gray'}
    fig, ax = plt.subplots(figsize=(16, 8))
    
    plot_func(df_periodo_gen, df_periodo_dem, periodo, ax, colores)
    
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    
    order = {label: i for i, label in enumerate(labels)}
    if 'Demanda' in order:
        demanda_handle = handles[order['Demanda']]
        demanda_label = labels[order['Demanda']]
        
        gen_handles = [h for i, h in enumerate(handles) if labels[i] != 'Demanda']
        gen_labels = [l for l in labels if l != 'Demanda']
        
        final_handles = [demanda_handle] + list(reversed(gen_handles))
        final_labels = [demanda_label] + list(reversed(gen_labels))
        
        ax.legend(final_handles, final_labels, title='Leyenda', bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax.legend(list(reversed(handles)), list(reversed(labels)), title='Tipo de Central', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        
    ruta_guardado = os.path.join(carpeta_salida, f'grafico_gen_dem_{periodo}.png')
    # plt.savefig(ruta_guardado, dpi=300)
    print(f"Gráfico guardado exitosamente en: {ruta_guardado}")
    plt.show()

# --- CONFIGURACIÓN Y EJECUCIÓN (SIN CAMBIOS) ---
if __name__ == "__main__":
    # --- ¡Elige el período que quieres analizar! ---
    PERIODO_SELECCIONADO = "2025-01-01"  # Vista detallada de un día
    # PERIODO_SELECCIONADO = "2025-06"       # Resumen de un mes
    # PERIODO_SELECCIONADO = "2025"         # Resumen de un año
    # PERIODO_SELECCIONADO = "2025-2035"     # Resumen de un rango de años
    
    RUTA_GENERACION_CSV = os.path.join("resultados_simulacion", "resultados_generacion.csv")
    RUTA_DEMANDA_CSV = os.path.join("resultados_simulacion", "resultados_demanda.csv")
    CARPETA_GRAFICOS = os.path.join("resultados_simulacion", "graficos")

    generar_grafico_generacion(PERIODO_SELECCIONADO, RUTA_GENERACION_CSV, RUTA_DEMANDA_CSV, CARPETA_GRAFICOS)