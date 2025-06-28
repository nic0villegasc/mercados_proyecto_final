import json
import pandas as pd
from pathlib import Path

class ProjectDataAPI:
    """
    Una API para cargar y acceder a todos los datos del proyecto de mercados eléctricos.
    Centraliza la lectura y el pre-procesamiento de los archivos de configuración y series de tiempo.
    """

    def __init__(self, data_path: str = 'data'):
        """
        Inicializa la API y carga todos los datos del proyecto.

        Args:
            data_path (str): La ruta a la carpeta principal de datos.
        """
        self.base_path = Path(data_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"El directorio de datos '{data_path}' no fue encontrado.")

        # Cargar datos de configuración
        self._load_config_data()

        # Cargar series de tiempo
        self._load_demand_data()
        self._load_centrales_timeseries_data()

        print("API de Datos inicializada y todos los datos han sido cargados.")

    def _load_config_data(self):
        """Carga los archivos de configuración JSON (centrales y líneas)."""
        # Cargar información de las centrales
        centrales_path = self.base_path / 'centrales.json'
        with open(centrales_path, 'r') as f:
            centrales_data = json.load(f)
        self.centrales = pd.DataFrame(centrales_data)

        # Cargar información de las líneas de transmisión
        lineas_path = self.base_path / 'lineas_transmision.json'
        with open(lineas_path, 'r') as f:
            lineas_data = json.load(f)
        self.lineas = pd.DataFrame(lineas_data)

    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Función de ayuda para limpiar columnas numéricas en los CSVs."""
        # Convierte a string, elimina espacios, quita comas y convierte a float
        return series.astype(str).str.strip().str.replace(',', '').astype(float)

    def _load_demand_data(self):
        """
        Carga los archivos de demanda y mantiene los IDs
        numéricos de las barras (1, 2, 3) de forma consistente.
        """
        all_demands_profiles = []
        # Usamos directamente los IDs numéricos de las barras
        barras_ids = [1, 2, 3] 

        for barra_id in barras_ids:
            try:
                demand_path = self.base_path / f'demanda{barra_id}.csv'
                df_profile = pd.read_csv(demand_path, dtype={'Hora': int})
                
                # Asignamos el ID numérico a la columna 'Barra'
                df_profile['Barra'] = barra_id
                df_profile['Demanda_MW'] = self._clean_numeric_column(df_profile['Demanda_MW'])
                all_demands_profiles.append(df_profile)
            except FileNotFoundError:
                print(f"Advertencia: No se encontró el archivo de demanda para la barra {barra_id}.")
                continue
        
        if not all_demands_profiles:
            self.demanda = pd.DataFrame()
            return

        demand_profiles_df = pd.concat(all_demands_profiles, ignore_index=True)

        # --- El resto de la función para crear el Timestamp no cambia, ---
        # --- pero ahora propagará los IDs numéricos. ---
        BASE_YEAR = 2025
        HORIZON_YEARS = 10
        start_date = pd.to_datetime(f'{BASE_YEAR}-01-01 00:00:00')
        end_date = pd.to_datetime(f'{BASE_YEAR + HORIZON_YEARS - 1}-12-31 23:00:00')
        time_index_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='h'), columns=['Timestamp'])
        
        # Creamos el scaffold usando los IDs numéricos
        barras_df = pd.DataFrame({'Barra': barras_ids})
        scaffold_df = time_index_df.merge(barras_df, how='cross')
        
        scaffold_df['Año'] = scaffold_df['Timestamp'].dt.year - BASE_YEAR + 1
        scaffold_df['Mes'] = scaffold_df['Timestamp'].dt.month
        scaffold_df['Hora'] = scaffold_df['Timestamp'].dt.hour
        
        self.demanda = pd.merge(
            scaffold_df,
            demand_profiles_df,
            on=['Año', 'Mes', 'Hora', 'Barra'],
            how='left'
        )
        
        if self.demanda['Demanda_MW'].isnull().any():
            print("Advertencia: Se rellenaron valores de demanda faltantes.")
            self.demanda['Demanda_MW'] = self.demanda['Demanda_MW'].ffill().ffill()
        

    def _load_centrales_timeseries_data(self):
        """
        Itera sobre el manifiesto de centrales y carga todas las series de tiempo
        asociadas (costos, generación, caudales, etc.).
        """
        self.time_series = {}
        for idx, central in self.centrales.iterrows():
            central_id = central['id_central']
            self.time_series[central_id] = {}

            # Lista de posibles rutas en el JSON
            rutas = {
                'generacion_fija': central.get('ruta_generacion_fija'),
                'costo_variable': central.get('ruta_costo_variable'),
                'gen_max_gestionable': central.get('ruta_gen_max_gestionable'),
                'caudales': central.get('rutas_caudales_afluentes')
            }

            # Cargar generación fija (solar)
            if pd.notna(rutas['generacion_fija']):
                df = pd.read_csv(rutas['generacion_fija'])
                df['Generacion_MW'] = self._clean_numeric_column(df['Generacion_MW'])
                self.time_series[central_id]['generacion_fija'] = df

            # Cargar costo variable (térmicas, diésel)
            if pd.notna(rutas['costo_variable']):
                df = pd.read_csv(rutas['costo_variable'])
                df['Costo_USD_MWh'] = self._clean_numeric_column(df['Costo_USD_MWh'])
                self.time_series[central_id]['costo_variable'] = df

            # Cargar generación máxima gestionable
            if pd.notna(rutas['gen_max_gestionable']):
                df = pd.read_csv(rutas['gen_max_gestionable'])
                # El nombre de la columna es GEN_MAX en el archivo
                df['Generacion_MW'] = self._clean_numeric_column(df['GEN_MAX'])
                self.time_series[central_id]['gen_max_gestionable'] = df.drop(columns=['GEN_MAX'])

            # Cargar caudales (hidroeléctricas)
            if isinstance(rutas['caudales'], dict):
                self.time_series[central_id]['caudales'] = {}
                for hidro_type, path in rutas['caudales'].items():
                    df = pd.read_csv(path)
                    df['Caudal_MWh'] = self._clean_numeric_column(df['Caudal_MWh'])
                    self.time_series[central_id]['caudales'][hidro_type] = df

    # --- FUNCIONES PÚBLICAS DE LA API ---

    def get_centrales_info(self) -> pd.DataFrame:
        """Devuelve un DataFrame con la información estática de todas las centrales."""
        return self.centrales

    def get_lineas_info(self) -> pd.DataFrame:
        """Devuelve un DataFrame con la información de las líneas de transmisión."""
        return self.lineas

    def get_demanda(self, barra: int = None) -> pd.DataFrame:
        """
        Devuelve la demanda horaria para 10 años.

        Args:
            barra (int, optional): Si se especifica, filtra la demanda para esa barra.

        Returns:
            pd.DataFrame: Un DataFrame con la demanda.
        """
        if barra:
            return self.demanda[self.demanda['Barra'] == barra].copy()
        return self.demanda.copy()

    def get_time_series_data(self, central_id: str, data_key: str) -> pd.DataFrame | dict:
        """
        Función genérica para obtener una serie de tiempo para una central específica.

        Args:
            central_id (str): El ID de la central (ej. "SOLAR-NORTE-01").
            data_key (str): La clave de los datos deseados (ej. "generacion_fija", "costo_variable", "caudales").

        Returns:
            pd.DataFrame or dict: El DataFrame con los datos, o un diccionario en el caso de los caudales.
        """
        if central_id not in self.time_series:
            raise ValueError(f"No se encontró la central con ID: {central_id}")
        if data_key not in self.time_series[central_id]:
             raise ValueError(f"La central '{central_id}' no tiene datos de tipo '{data_key}'.")
        return self.time_series[central_id][data_key]

    def get_caudal(self, central_id: str, hidrologia: str = 'media') -> pd.DataFrame:
        """
        Obtiene el caudal para una central hidroeléctrica y una hidrología específicas.

        Args:
            central_id (str): El ID de la central hidroeléctrica.
            hidrologia (str): El tipo de hidrología ('seca_p10', 'media', 'humeda_p90').

        Returns:
            pd.DataFrame: El DataFrame con los caudales mensuales.
        """
        caudales_dict = self.get_time_series_data(central_id, 'caudales')
        if hidrologia not in caudales_dict:
            raise ValueError(f"Hidrología '{hidrologia}' no válida. Opciones: {list(caudales_dict.keys())}")
        return caudales_dict[hidrologia]

# --- Ejemplo de Uso ---
if __name__ == '__main__':
    api = ProjectDataAPI()

    # 2. Obtener información general
    print("\n--- Información de Centrales ---")
    info_centrales = api.get_centrales_info()
    print(info_centrales)

    print("\n--- Información de Líneas de Transmisión ---")
    info_lineas = api.get_lineas_info()
    print(info_lineas)

    # 3. Obtener series de tiempo
    print("\n--- Demanda de las Barras (primeras 5 filas) ---")
    demanda_barra_2 = api.get_demanda()
    print(demanda_barra_2.head())

    print("\n--- Costo Variable de la Térmica del Norte (primeras 5 filas) ---")
    costo_termica_norte = api.get_time_series_data('TERMICA-CENTRO-02', 'gen_max_gestionable')
    print(costo_termica_norte)

    print("\n--- Caudal de la Hidro del Sur en hidrología seca (primeras 5 filas) ---")
    caudal_hidro_sur_seca = api.get_caudal('HIDRO-SUR-03', hidrologia='seca_p10')
    print(caudal_hidro_sur_seca.head())