import os
import json
from pathlib import Path
import pandas as pd
from data_api import ProjectDataAPI
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo

class ModelData:
    """
    Esta clase utiliza una instancia de ProjectDataAPI para construir
    los conjuntos e índices necesarios para el modelo de optimización de Pyomo.
    """
    def __init__(self, data_api: ProjectDataAPI, hidrologia: str = 'media', months_to_run: int = None):
        """
        Inicializa y construye los conjuntos a partir de la API de datos.
        """
        self.api = data_api
        self.months_to_run = months_to_run
        self.hidrologia = hidrologia
        self._create_all_sets()
        print(f"Paso 1 completado: Usando la hidrología '{self.hidrologia}'.")
        self._create_parameters()
       
    def _create_all_sets(self):
        """Método principal para orquestar la creación de todos los conjuntos."""
        centrales_df = self.api.get_centrales_info()
        lineas_df = self.api.get_lineas_info()
        demanda_df = self.api.get_demanda()
        
        if self.months_to_run is not None:
            print(f"MODO DE PRUEBA: Filtrando datos para los primeros {self.months_to_run} mes(es).")
            unique_months = sorted(demanda_df['Timestamp'].dt.to_period('M').unique())
            if self.months_to_run > len(unique_months):
                self.months_to_run = len(unique_months)
            
            last_month_to_include = unique_months[self.months_to_run - 1]
            # Filtramos el DataFrame para que solo contenga datos hasta el final de ese mes
            demanda_df = demanda_df[demanda_df['Timestamp'].dt.to_period('M') <= last_month_to_include]

        # 1. Definir Conjuntos de Generadores
        self.G = centrales_df['id_central'].tolist()
        self.G_termica = centrales_df[centrales_df['tecnologia'] == 'termica']['id_central'].tolist()
        self.G_diesel = centrales_df[centrales_df['tecnologia'] == 'diesel']['id_central'].tolist()
        self.G_solar = centrales_df[centrales_df['tecnologia'] == 'solar']['id_central'].tolist()
        self.G_hidro = centrales_df[centrales_df['tecnologia'] == 'hidroelectrica']['id_central'].tolist()

        # 2. Definir Conjuntos de la Red
        self.L = lineas_df['id_linea'].tolist()
        barras_centrales = set(centrales_df['id_barra'].astype(int).unique())
        barras_lineas_origen = set(lineas_df['barra_origen'].astype(int).unique())
        barras_lineas_destino = set(lineas_df['barra_destino'].astype(int).unique())
        barras_demanda = set(demanda_df['Barra'].astype(int).unique())
        
        self.B = sorted(list(barras_centrales | barras_lineas_origen | barras_lineas_destino | barras_demanda))

        
        # 3. Definir Conjuntos de Tiempo (usando el DataFrame de demanda de la API)
        if 'Timestamp' not in demanda_df.columns:
            raise ValueError("El DataFrame de demanda de la API no contiene la columna 'Timestamp' requerida.")
        unique_timestamps = sorted(demanda_df['Timestamp'].unique())
        
        self.T = list(range(1, len(unique_timestamps) + 1))

        self.hourly_mapping_num_to_ts = {i + 1: ts for i, ts in enumerate(unique_timestamps)}
        self.hourly_mapping_ts_to_num = {ts: i + 1 for i, ts in enumerate(unique_timestamps)}

        df_timestamps = pd.DataFrame(unique_timestamps, columns=['Timestamp'])

        df_timestamps['t_index'] = df_timestamps['Timestamp'].map(self.hourly_mapping_ts_to_num)

        df_timestamps['mes_absoluto'] = df_timestamps['Timestamp'].dt.year * 100 + df_timestamps['Timestamp'].dt.month
        mapa_mes_absoluto_a_relativo = {
            mes_abs: i + 1 for i, mes_abs in enumerate(sorted(df_timestamps['mes_absoluto'].unique()))
        }
        
        df_timestamps['m_index'] = df_timestamps['mes_absoluto'].map(mapa_mes_absoluto_a_relativo)
        self.M = sorted(list(mapa_mes_absoluto_a_relativo.values()))
        self.T_m = df_timestamps.groupby('m_index')['t_index'].apply(list).to_dict()
        
        # 4. Definir Conjuntos de Conectividad
        self.G_b = centrales_df.groupby('id_barra')['id_central'].apply(list).to_dict()
        self.L_out = lineas_df.groupby('barra_origen')['id_linea'].apply(list).to_dict()
        self.L_in = lineas_df.groupby('barra_destino')['id_linea'].apply(list).to_dict()
        self.t_to_m_map = {t: m for m, t_list in self.T_m.items() for t in t_list}
        
        for barra in self.B:
            self.G_b.setdefault(barra, [])
            self.L_out.setdefault(barra, [])
            self.L_in.setdefault(barra, [])

    def _create_parameters(self):
      """
      Crea los diccionarios de parámetros que el modelo de Pyomo necesita,
      utilizando los datos de la API.
      """
      print("\nPaso 2: Creando los parámetros del modelo...")
      
      # --- Parámetro: Costo Variable ---
      self.CostoVar = {}
      
      # Unimos las listas de generadores térmicos y diésel, que son los que tienen costo variable
      generadores_con_costo = self.G_termica + self.G_diesel
      
      for g in generadores_con_costo:
        # 1. Obtenemos el DataFrame de costos para el generador 'g'
        costo_df = self.api.get_time_series_data(g, 'costo_variable')
        
        # 2. Creamos un mapeo eficiente de (Año, Mes) -> Costo
        # Esto evita buscar en el DataFrame para cada una de las 87,600 horas.
        costo_map = costo_df.set_index(['Año', 'Mes'])['Costo_USD_MWh'].to_dict()
        
        # 3. Iteramos sobre TODAS las horas 't' del horizonte de simulación
        for t in self.T:
            # Obtenemos el timestamp (fecha y hora exactas) para la hora numérica 't'
            timestamp = self.hourly_mapping_num_to_ts[t]
            
            # Calculamos el año relativo (1-10) y el mes (1-12) para ese timestamp
            año_relativo = timestamp.year - 2025 + 1
            mes = timestamp.month
            
            # Buscamos el costo en nuestro mapeo
            costo_hora = costo_map.get((año_relativo, mes))
            
            if costo_hora is not None:
                # Creamos la entrada en el diccionario para Pyomo: (generador, hora) -> costo
                self.CostoVar[(g, t)] = costo_hora
            else:
                # Opcional: Manejar casos donde una hora no tenga un costo definido
                print(f"Advertencia: No se encontró costo para {g} en la hora {t} (Año {año_relativo}, Mes {mes})")
      
      # --- Parámetros: Límites de Generación ---
      self.Pmin = {}
      # Pmax_inst: Potencia MÁXIMA INSTANTÁNEA que una central puede generar (MW)
      # Es un límite físico de la turbina/generador.
      self.Pmax_inst = {}

      # EnergiaMaxMensual: Límite de ENERGÍA TOTAL que una central puede generar en un mes (MWh)
      # Es un presupuesto mensual.
      self.EnergiaMaxMensual = {}

      centrales_df = self.api.get_centrales_info()
      centrales_info_map = centrales_df.set_index('id_central').to_dict('index')

      # Generadores que tienen un presupuesto de energía mensual
      generadores_gestionables = self.G_termica + self.G_diesel + self.G_hidro

      for g in self.G:
          info_gen = centrales_info_map.get(g, {})
          
          # 1. Asignar Potencia Mínima (Pmin)
          pmin_val = info_gen.get('minimo_tecnico_mw', 0)
          self.Pmin[g] = pmin_val

          # 2. Asignar Potencia Máxima INSTANTÁNEA (Pmax_inst)
          # Este es el límite físico en MW en cualquier hora t.
          # CORRECCIÓN: Se usa 'capacidad_nominal_mw' según lo indicado.
          pmax_inst_val = info_gen.get('capacidad_nominal_mw', 0)
          self.Pmax_inst[g] = pmax_inst_val

      # 3. Asignar el Presupuesto de ENERGÍA MENSUAL para centrales gestionables
      for g in generadores_gestionables:
          try:
              # La API nos da la serie de tiempo con el presupuesto mensual
              energia_max_df = self.api.get_time_series_data(g, 'gen_max_gestionable')
              
              # Creamos un mapa (Año, Mes) -> Energia_MWh para búsqueda rápida
              energia_map = energia_max_df.set_index(['Año', 'Mes'])['Generacion_MW'].to_dict()

              # Iteramos sobre los MESES del horizonte (m de 1 a 120)
              for m in self.M:
                  # Obtenemos el año relativo y mes del calendario para el mes 'm'
                  primera_hora_del_mes = self.T_m[m][0]
                  timestamp = self.hourly_mapping_num_to_ts[primera_hora_del_mes]
                  año_relativo = timestamp.year - 2025 + 1
                  mes_calendario = timestamp.month
                  
                  # Buscamos el presupuesto de energía para ese mes
                  presupuesto_energia = energia_map.get((año_relativo, mes_calendario))

                  if presupuesto_energia is not None:
                      # La clave es (generador, mes)
                      self.EnergiaMaxMensual[(g, m)] = presupuesto_energia

          except ValueError as e:
              print(f"Advertencia al buscar gen_max_gestionable para {g}: {e}")

      print(f"Parámetro 'Pmin' creado con {len(self.Pmin)} entradas.")
      print(f"Parámetro 'Pmax_inst' (límite horario MW) creado con {len(self.Pmax_inst)} entradas.")
      print(f"Parámetro 'EnergiaMaxMensual' (presupuesto MWh) creado con {len(self.EnergiaMaxMensual)} entradas.")
      
      # --- Parámetro: Demanda Horaria por Barra ---
      self.Demanda = {}
      demanda_df = self.api.get_demanda()

      if not demanda_df.empty:
          # Mapeamos la columna 'Timestamp' a nuestro índice horario numérico 't'
          # para usarlo como clave en el diccionario.
          demanda_df['t_index'] = demanda_df['Timestamp'].map(self.hourly_mapping_ts_to_num)

          # Creamos el diccionario para Pyomo de la forma más eficiente:
          # El índice del DataFrame se convierte en las claves del diccionario.
          # La clave será una tupla: (Barra, t_index)
          self.Demanda = demanda_df.set_index(['Barra', 't_index'])['Demanda_MW'].to_dict()

      print(f"Parámetro 'Demanda' creado con {len(self.Demanda)} entradas.")
      
      # --- Parámetros: Red de Transmisión ---
      self.FlujoMax = {}
      lineas_df = self.api.get_lineas_info()

      if not lineas_df.empty:
          # Creamos un diccionario con el formato {id_linea: capacidad_mw}
          # La forma más directa es usar set_index y to_dict
          if 'capacidad_mw' in lineas_df.columns:
              self.FlujoMax = lineas_df.set_index('id_linea')['capacidad_mw'].to_dict()
          else:
              print("Advertencia: No se encontró la columna 'capacidad_mw' en los datos de las líneas.")

      print(f"Parámetro 'FlujoMax' creado con {len(self.FlujoMax)} entradas.")
      
      # --- Parámetro: Potencia Solar Disponible ---
      self.PdispSolar = {}

      for g in self.G_solar:
          try:
              # 1. Obtenemos el DataFrame con la serie de tiempo horaria ya procesada.
              #    Columnas: ['Timestamp', 'Generacion_MW']
              solar_df = self.api.get_solar_generacion_fija(g)

              # 2. Creamos un mapeo eficiente usando el Timestamp como índice.
              #    Esto convierte el DataFrame en una Serie de pandas donde el índice es la fecha
              #    y el valor es la generación. Es la forma más rápida de buscar por fecha.
              generation_series = solar_df.set_index('Timestamp')['Generacion_MW']

              # 3. Iteramos sobre TODAS las horas 't' del horizonte de simulación.
              for t in self.T:
                  # Obtenemos el objeto Timestamp correspondiente a la hora 't'.
                  timestamp = self.hourly_mapping_num_to_ts[t]

                  # 4. Buscamos la generación directamente con el Timestamp en la Serie.
                  #    Usamos .get() con un valor por defecto de 0 para manejar de forma segura
                  #    cualquier posible fecha que no se encuentre.
                  generacion_hora = generation_series.get(timestamp, default=0)

                  # 5. Creamos la entrada en el diccionario para Pyomo: (generador, hora) -> generacion
                  self.PdispSolar[(g, t)] = generacion_hora

          except (ValueError, KeyError) as e:
              print(f"Advertencia al procesar la generación para {g}: {e}")

      print(f"Parámetro 'PdispSolar' creado con {len(self.PdispSolar)} entradas.")
      
      # --- Parámetros para el Balance Hídrico ---
      self.Afluente = {}
      self.CotaInicial = {}
      self.CotaMaxima = {}
      
      # --- Cálculo del Valor Terminal del Agua (dependiente del escenario) ---
      self.ValorTerminalAgua = {}
      
      # 1. Calcular el costo térmico promedio como valor base.
      if self.CostoVar:
          costo_promedio_termico = sum(self.CostoVar.values()) / len(self.CostoVar)
      else:
          costo_promedio_termico = 60.0 # Un valor por defecto si no hay centrales térmicas.
      
      # 2. Ajustar el valor terminal usando una heurística basada en el escenario.
      #    self.hidrologia se define al crear el objeto ModelData.
      if self.hidrologia == 'P10': # Escenario SECO
          # En sequía, el agua futura es más valiosa para evitar usar diésel caro.
          VALOR_TERMINAL_CONSTANTE = 1.5 * costo_promedio_termico
          print(f"Hidrología SECA (P10) detectada. Usando Valor Terminal ALTO: {VALOR_TERMINAL_CONSTANTE:.2f} USD/MWh")
      
      elif self.hidrologia == 'P90': # Escenario HÚMEDO
          # En abundancia, el agua futura tiene menos valor marginal.
          VALOR_TERMINAL_CONSTANTE = 0.8 * costo_promedio_termico
          print(f"Hidrología HÚMEDA (P90) detectada. Usando Valor Terminal BAJO: {VALOR_TERMINAL_CONSTANTE:.2f} USD/MWh")
      
      else: # Escenario MEDIA o por defecto
          VALOR_TERMINAL_CONSTANTE = costo_promedio_termico
          print(f"Hidrología MEDIA detectada. Usando Valor Terminal BASE: {VALOR_TERMINAL_CONSTANTE:.2f} USD/MWh")

      # 3. Asignar el valor terminal calculado a cada central hidroeléctrica.
      for g in self.G_hidro:
          self.ValorTerminalAgua[g] = VALOR_TERMINAL_CONSTANTE
      
      centrales_df = self.api.get_centrales_info()
      
      for g in self.G_hidro:
          # 1. Guardar la Cota Inicial para cada central hidroeléctrica
          info_hidro = centrales_df[centrales_df['id_central'] == g].iloc[0]
          self.CotaInicial[g] = info_hidro['cota_inicial_mwh']
          self.CotaMaxima[g] = info_hidro['cota_maxima_mwh']
          
          self.ValorTerminalAgua[g] = VALOR_TERMINAL_CONSTANTE
          
          # 2. Calcular el afluente horario a partir de los datos mensuales
          caudal_df = self.api.get_caudal(g, hidrologia='media')
          caudal_map = caudal_df.set_index(['Año', 'Mes'])['Caudal_MWh'].to_dict()
          
          # Iteramos sobre cada mes del horizonte de simulación
          for m in self.M:
              horas_del_mes = self.T_m.get(m, [])
              if not horas_del_mes:
                  continue
              
              num_horas_mes = len(horas_del_mes)
              
              # Obtenemos el año y mes para buscar el caudal
              timestamp_mes = self.hourly_mapping_num_to_ts[horas_del_mes[0]]
              año_relativo = timestamp_mes.year - 2025 + 1
              mes_calendario = timestamp_mes.month
              
              caudal_mensual = caudal_map.get((año_relativo, mes_calendario), 0)
              
              # Distribuimos el caudal mensual uniformemente en cada hora del mes
              afluente_horario = caudal_mensual / num_horas_mes
              
              # Asignamos el mismo afluente horario a todas las horas de este mes
              for t in horas_del_mes:
                  self.Afluente[(g, t)] = afluente_horario
                  
      print(f"Parámetro 'Afluente' (horario) creado con {len(self.Afluente)} entradas.")
      print(f"Parámetro 'CotaInicial' creado con {len(self.CotaInicial)} entradas.")

class OptimizationModel:
    """
    Construye el modelo de optimización en Pyomo utilizando los datos preparados.
    """
    def __init__(self, model_data: ModelData):
        """
        Inicializa y construye el modelo de Pyomo con sus variables.

        Args:
            model_data (ModelData): El objeto que contiene todos los conjuntos y parámetros.
        """
        self.data = model_data
        self.model = pyo.ConcreteModel()
        
        self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        
        # Llamamos al método que construye las variables
        self._build_variables()
        
        print("\nPaso 3: Variables de decisión construidas.")
        
        self._build_objective()
        
        self._build_constraints()

    def _build_variables(self):
        """
        Define todas las variables de decisión del modelo.
        """
        # --- p(g, t): Potencia despachada (generada) por la unidad g en la hora t [MW] ---
        # pyo.Var crea una variable.
        # El primer argumento son los conjuntos que la indexan: (Generadores, Horas).
        # El dominio 'NonNegativeReals' asegura que la variable sea continua y >= 0.
        self.model.p = pyo.Var(self.data.G, self.data.T, domain=pyo.NonNegativeReals)
        
        # --- u(g, m): Estado de encendido de la central térmica g en el mes m ---
        # El conjunto de indexación es solo para las centrales térmicas.
        generadores_unit_commitment = self.data.G_termica
        # El dominio 'Binary' asegura que la variable solo puede ser 0 o 1.
        self.model.u = pyo.Var(generadores_unit_commitment, self.data.M, domain=pyo.Binary)
        
        # --- p_vert(g, t): Potencia solar no inyectada (vertimiento) por la unidad g en la hora t [MW] ---
        # Se define solo para las centrales solares.
        # El dominio es NonNegativeReals porque no se puede "verter negativamente".
        self.model.p_vert = pyo.Var(self.data.G_solar, self.data.T, domain=pyo.NonNegativeReals)
        
        # --- f(l, t): Flujo de potencia en la línea l en la hora t [MW] ---
        # El dominio es 'Reals' porque el flujo puede ser positivo o negativo,
        # indicando la dirección del flujo.
        self.model.f = pyo.Var(self.data.L, self.data.T, domain=pyo.Reals)
        
        # --- v(g, t): Volumen de agua en el embalse g al final de la hora t [MWh] ---
        # Se define solo para las centrales hidroeléctricas.
        # El dominio es NonNegativeReals, ya que el volumen no puede ser negativo.
        self.model.v = pyo.Var(self.data.G_hidro, self.data.T, domain=pyo.NonNegativeReals)
        
        # --- s(g, t): Agua vertida (derramada) desde el embalse g en la hora t [MWh] ---
        # Se define para las centrales hidroeléctricas. Permite al modelo "desechar"
        # agua si es necesario (ej. por exceso de afluentes).
        self.model.s = pyo.Var(self.data.G_hidro, self.data.T, domain=pyo.NonNegativeReals)

    def _build_objective(self):
        """
        Define la función objetivo del modelo: minimizar el costo total de operación,
        maximizando implícitamente el valor del agua remanente.
        """
        print("Construyendo la función objetivo final...")
        
        # 1. Suma de los costos de combustible de las centrales térmicas y diésel.
        #    (Esto es lo que queremos minimizar)
        generadores_costo_combustible = self.data.G_termica + self.data.G_diesel
        costo_combustible = sum(self.data.CostoVar[g, t] * self.model.p[g, t] 
                                for g in generadores_costo_combustible 
                                for t in self.data.T 
                                if (g, t) in self.data.CostoVar)
        
        # 2. Suma del valor del agua que queda en los embalses en la última hora.
        #    (Esto es lo que queremos maximizar)
        T_final = self.data.T[-1] # Obtenemos el índice de la última hora de la simulación
        valor_agua_remanente = sum(self.data.ValorTerminalAgua[g] * self.model.v[g, T_final] 
                                   for g in self.data.G_hidro)
        
        # 3. La función objetivo final.
        #    Minimizar (Costos) es igual a Minimizar (Costos - Ingresos/Beneficios).
        #    Al restar el valor del agua remanente, el modelo es incentivado a
        #    maximizar ese valor para minimizar el resultado total.
        costo_total_neto = costo_combustible - valor_agua_remanente
        
        self.model.objective = pyo.Objective(expr=costo_total_neto, sense=pyo.minimize)
        
        print("Función objetivo final construida exitosamente.")
        
    def _build_constraints(self):
        """Define todas las restricciones del modelo."""
        print("Construyendo las restricciones...")
        
        # --- 1. Balance de Potencia en cada Barra (para cada b, t) ---
        def power_balance_rule(model, b, t):
            # Lado izquierdo: Todo lo que entra a la barra
            generacion_en_barra = sum(model.p[g, t] for g in self.data.G_b.get(b, []))
            flujo_entrante = sum(model.f[l, t] for l in self.data.L_in.get(b, []))
            
            # Lado derecho: Todo lo que sale de la barra
            demanda_en_barra = self.data.Demanda.get((b, t), 0)
            flujo_saliente = sum(model.f[l, t] for l in self.data.L_out.get(b, []))
            
            # La ecuación: (Generación + Flujo Entrante) == (Demanda + Flujo Saliente)
            return generacion_en_barra + flujo_entrante == demanda_en_barra + flujo_saliente

        self.model.power_balance_constr = pyo.Constraint(self.data.B, self.data.T, rule=power_balance_rule)
        print("OK: Restricción 'Balance de Potencia' construida.")
        
         # --- 2. Límites de Capacidad de las Líneas (para cada l, t) ---
        def line_capacity_rule(model, l, t):
            flujo_max = self.data.FlujoMax.get(l)
            # Si una línea no tiene capacidad definida, no se aplica la restricción
            if flujo_max is None:
                return pyo.Constraint.Skip
            return pyo.inequality(-flujo_max, model.f[l, t], flujo_max)

        self.model.line_capacity_constr = pyo.Constraint(self.data.L, self.data.T, rule=line_capacity_rule)
        print("OK: Restricción 'Límites de Capacidad de Líneas' construida.")
        
        # --- 3.a Límite SUPERIOR de Generación de Centrales Térmicas ---
        def thermal_pmax_rule(model, g, t):
            m = self.data.t_to_m_map.get(t)
            if m is None: return pyo.Constraint.Skip
            upper_bound = self.data.Pmax_inst.get(g, 0) * model.u[g, m]
            return model.p[g, t] <= upper_bound
        self.model.thermal_pmax_constr = pyo.Constraint(self.data.G_termica, self.data.T, rule=thermal_pmax_rule)
        print("OK: Restricción 'Límite Superior de Generación Térmica' construida.")

        # --- 3.b Límite INFERIOR de Generación de Centrales Térmicas ---
        def thermal_pmin_rule(model, g, t):
            m = self.data.t_to_m_map.get(t)
            if m is None: return pyo.Constraint.Skip
            lower_bound = self.data.Pmin.get(g, 0) * model.u[g, m]
            return model.p[g, t] >= lower_bound
        self.model.thermal_pmin_constr = pyo.Constraint(self.data.G_termica, self.data.T, rule=thermal_pmin_rule)
        print("OK: Restricción 'Límite Inferior de Generación Térmica' construida.")
        
        # --- 4. Balance de Generación Solar (para cada g solar, t) ---
        def solar_balance_rule(model, g, t):
            # La generación despachada (p) más el vertimiento (p_vert) debe
            # ser igual a la potencia solar disponible en esa hora.
            potencia_disponible = self.data.PdispSolar.get((g, t), 0)
            return model.p[g, t] + model.p_vert[g, t] == potencia_disponible
        
        self.model.solar_balance_constr = pyo.Constraint(self.data.G_solar, self.data.T, rule=solar_balance_rule)
        print("OK: Restricción 'Balance de Generación Solar' construida.")
        
        # --- 5. Límites de Generación para Centrales FLEXIBLES (Hidro y Diésel) ---
        generadores_flexibles = self.data.G_hidro + self.data.G_diesel
        def flexible_generation_limits_rule(model, g, t):
            # La generación horaria no puede superar la capacidad máxima instantánea.
            upper_bound = self.data.Pmax_inst.get(g, 0)
            # Para estas centrales, el mínimo es siempre 0.
            return model.p[g, t] <= upper_bound

        self.model.flexible_generation_limits_constr = pyo.Constraint(generadores_flexibles, self.data.T, rule=flexible_generation_limits_rule)
        print("OK: Restricción 'Límites de Generación Flexible (Hidro/Diésel)' construida.")
        
        # --- 6. Presupuesto Mensual de Energía para Centrales Gestionables ---
        generadores_gestionables = self.data.G_termica + self.data.G_diesel + self.data.G_hidro
        def monthly_energy_budget_rule(model, g, m):
            # La suma de la generación horaria en un mes no debe superar el presupuesto mensual
            energia_generada_mes = sum(model.p[g, t] for t in self.data.T_m.get(m, []))
            presupuesto_energia = self.data.EnergiaMaxMensual.get((g, m))
            if presupuesto_energia is None:
                return pyo.Constraint.Skip
            return energia_generada_mes <= presupuesto_energia
        self.model.monthly_energy_budget_constr = pyo.Constraint(generadores_gestionables, self.data.M, rule=monthly_energy_budget_rule)
        print("OK: Restricción 'Presupuesto Mensual de Energía' construida.")

        def solar_cap_limit_rule(model, t):
            return sum(model.p[g, t] for g in self.data.G_solar) <= 50
        self.model.solar_cap_limit = pyo.Constraint(self.data.T,
                                                    rule=solar_cap_limit_rule)
        print("OK: Restricción 'Solar inyectada ≤ 50 MW' construida.")
        
        # --- 7. Balance Hídrico para Embalses Hidroeléctricos ---
        def water_balance_rule(model, g, t):
            # Condición para la primera hora de la simulación (t=1)
            # El volumen inicial es la Cota Inicial del embalse.
            # Nota: self.data.T[0] es la primera hora, que en este modelo es 1.
            if t == self.data.T[0]:
                return model.v[g, t] == (self.data.CotaInicial[g] 
                                         + self.data.Afluente.get((g, t), 0)
                                         - model.p[g, t] 
                                         - model.s[g, t])
            
            # Condición para todas las demás horas (t > 1)
            # El volumen inicial es el volumen final de la hora anterior.
            else:
                return model.v[g, t] == (model.v[g, t-1] 
                                         + self.data.Afluente.get((g, t), 0)
                                         - model.p[g, t] 
                                         - model.s[g, t])

        self.model.water_balance_constr = pyo.Constraint(self.data.G_hidro, 
                                                         self.data.T, 
                                                         rule=water_balance_rule)
        print("OK: Restricción 'Balance Hídrico' construida.")
        
        
        # --- 9. Límite de Generación por Agua Disponible ---
        def hydro_generation_water_limit_rule(model, g, t):
            # Para la primera hora, la generación está limitada por la cota inicial más el primer afluente.
            if t == self.data.T[0]:
                agua_disponible = self.data.CotaInicial[g] + self.data.Afluente.get((g, t), 0)
                return model.p[g, t] <= agua_disponible
            
            # Para las demás horas, la generación está limitada por el volumen de la hora anterior más el afluente actual.
            else:
                agua_disponible = model.v[g, t-1] + self.data.Afluente.get((g, t), 0)
                return model.p[g, t] <= agua_disponible
        
        self.model.hydro_generation_water_limit_constr = pyo.Constraint(self.data.G_hidro, 
                                                                        self.data.T, 
                                                                        rule=hydro_generation_water_limit_rule)
        print("OK: Restricción 'Límite de Generación por Agua Disponible' construida.")
        
        
        def reservoir_max_limit_rule(model, g, t):
            # El volumen en cualquier hora t no puede superar la cota máxima del embalse.
            return model.v[g, t] <= self.data.CotaMaxima[g]

        self.model.reservoir_max_limit_constr = pyo.Constraint(self.data.G_hidro, 
                                                               self.data.T, 
                                                               rule=reservoir_max_limit_rule)
        print("OK: Restricción 'Límites de Capacidad del Embalse' construida.")


    def solve(self, solver_name='cplex'):
        """
        Resuelve el modelo utilizando el solver especificado.

        Args:
            solver_name (str): El nombre del solver a utilizar (ej. 'cplex', 'gurobi').
        """
        print(f"\n--- PASO 6: RESOLVIENDO EL MODELO CON {solver_name.upper()} ---")
        
        # 1. Crear una instancia del solver
        # Pyomo buscará el ejecutable del solver en el PATH del sistema.
        solver = pyo.SolverFactory(solver_name)
        
        # 2. Configurar opciones del solver (opcional)
        # Por ejemplo, establecer un límite de tiempo de 10 minutos (600 segundos)
        # solver.options['timelimit'] = 600

        # 3. Resolver el modelo
        # tee=True muestra el log del solver en la consola, lo que es crucial
        # para ver el progreso en problemas largos.
        self.results = solver.solve(self.model, tee=True)
        
    def solve_and_get_duals(self, solver_name='cplex'):
        """
        Resuelve el modelo en dos etapas para obtener los costos marginales (duales).
        """
        solver = pyo.SolverFactory(solver_name)
        
        # --- ETAPA 1: Resolver el MILP para obtener las decisiones de encendido (u) ---
        print(f"\n--- ETAPA 1: Resolviendo el MILP con {solver_name.upper()} ---")
        milp_results = solver.solve(self.model, tee=True)
        
        if milp_results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(f"La Etapa 1 (MILP) no encontró una solución óptima. Estado: {milp_results.solver.termination_condition}")
            self.results = milp_results
            return
            
        print("\nEtapa 1 completada. Decisiones de encendido encontradas.")
        
        # --- ETAPA 2: Fijar las variables binarias y resolver el LP para obtener los duales ---
        print("\n--- ETAPA 2: Fijando variables enteras y resolviendo el LP ---")
        
        # Fijamos las variables 'u' a su valor óptimo
        for g in self.data.G_termica:
            for m in self.data.M:
                self.model.u[g,m].fix(pyo.value(self.model.u[g,m]))
        
        # Resolvemos el problema de nuevo, que ahora es un LP
        self.results = solver.solve(self.model, tee=True)

        # Después de resolver el LP, los duales estarán disponibles
        if self.results.solver.status == pyo.SolverStatus.ok and self.results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("\n¡Solución óptima del LP encontrada! Los costos marginales están disponibles.")
        else:
            print(f"La Etapa 2 (LP) no encontró una solución óptima. Estado: {self.results.solver.termination_condition}")
    
    def export_model_to_file(self, filename="modelo.lp"):
        """
        Guarda la formulación matemática del modelo en un archivo.

        Args:
            filename (str): Nombre del archivo (ej. 'modelo.lp' o 'modelo.sav')
        """
        print(f"\nExportando el modelo al archivo: {filename}...")
        # La opción 'symbolic_solver_labels' es crucial para que los nombres de las
        # variables y restricciones en el archivo sean legibles.
        self.model.write("model.lp", io_options={'symbolic_solver_labels': True})
        print("Exportación completada.")
    
    def _add_marginal_generator_info(self, results_dfs):
        """
        Enriquece el DataFrame de costos marginales con la central que fija el precio.
        Utiliza copias explícitas de los DataFrames para evitar efectos secundarios.
        """
        print("Identificando la central marginal para cada paso horario...")
        
        cml_df = results_dfs['costo_marginal'].copy()
        gen_df = results_dfs['generacion'].copy()

        # Crear una tabla de costos de cada generador en cada hora
        costos_data = []
        for g in self.data.G:
            for t in self.data.T:
                # El costo de operación es el CostoVariable, o 0 si no tiene (hidro/solar).
                costo = self.data.CostoVar.get((g, t), 0)
                costos_data.append({'generador': g, 'paso_horario': t, 'costo_operacion': costo})
        costos_df = pd.DataFrame(costos_data)
        
        # Unir los resultados de generación con los costos de operación
        gen_con_costos = pd.merge(gen_df, costos_df, on=['generador', 'paso_horario'])
        
        # Unir costos marginales con la generación y sus costos
        merged_df = pd.merge(cml_df, gen_con_costos, on='paso_horario', how='left')
        
        merged_df.reset_index(drop=True, inplace=True)

        # Encontrar la central marginal
        centrales_info = self.data.api.get_centrales_info().set_index('id_central')
        merged_df['barra_gen'] = merged_df['generador'].map(centrales_info['id_barra'])
        
        merged_df['diff_costo'] = abs(merged_df['costo_marginal_usd_mwh'] - merged_df['costo_operacion'])
        
        # Para cada barra y paso horario, encontramos la central con la mínima diferencia
        idx = merged_df.groupby(['barra', 'paso_horario'])['diff_costo'].idxmin()
        centrales_marginales_df = merged_df.loc[idx][['barra', 'paso_horario', 'generador']]
        centrales_marginales_df.rename(columns={'generador': 'central_marginal'}, inplace=True)
        
        # Unimos esta información al DataFrame original de costos marginales
        cml_df_final = pd.merge(cml_df, centrales_marginales_df, on=['barra', 'paso_horario'], how='left')
        results_dfs['costo_marginal'] = cml_df_final.fillna('Congestion/Solar')
        
        return results_dfs
    
    def get_results_as_dataframes(self):
        """
        Extrae los resultados de las variables del modelo resuelto y los
        convierte en DataFrames de pandas para un fácil análisis, incluyendo
        columnas de año, mes y hora (0-23).
        
        Returns:
            dict: Un diccionario de DataFrames, con una clave por cada tipo de variable.
        """
        if self.results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("No se puede extraer resultados porque no se encontró una solución óptima.")
            return None

        print("\n--- PASO 7: EXtrayendo resultados a DataFrames ---")
        
        results_dfs = {}
        
        # Generación horaria (p)
        p_data = []
        for (g, t), var in self.model.p.items():
            if pyo.value(var, exception=False) > 1e-6:
                # Usamos 'paso_horario' como identificador único de tiempo
                p_data.append({'generador': g, 'paso_horario': t, 'generacion_mw': pyo.value(var)})
        results_dfs['generacion'] = pd.DataFrame(p_data)
        
        # Estado de encendido mensual (u) - No tiene dimensión horaria, no se modifica
        u_data = []
        for (g, m), var in self.model.u.items():
            u_data.append({'generador': g, 'mes': m, 'encendido': int(pyo.value(var))})
        results_dfs['compromiso_unidad'] = pd.DataFrame(u_data)

        # Vertimiento solar (p_vert)
        vert_data = []
        for (g, t), var in self.model.p_vert.items():
            if pyo.value(var, exception=False) > 1e-6:
                vert_data.append({'generador': g, 'paso_horario': t, 'vertimiento_mw': pyo.value(var)})
        results_dfs['vertimiento'] = pd.DataFrame(vert_data)

        # Flujo en líneas (f)
        f_data = []
        for (l, t), var in self.model.f.items():
            flow_val = pyo.value(var, exception=False)
            if abs(flow_val) > 1e-6:
                f_data.append({'linea': l, 'paso_horario': t, 'flujo_mw': flow_val})
        results_dfs['flujo'] = pd.DataFrame(f_data)
        
        # Nivel del embalse horario (v)
        v_data = []
        for (g, t), var in self.model.v.items():
            # Guardamos el nivel del embalse, incluso si es cercano a cero.
            vol_val = pyo.value(var, exception=False)
            if vol_val is not None:
                v_data.append({'generador': g, 'paso_horario': t, 'nivel_embalse_mwh': vol_val})
        results_dfs['nivel_embalse_horario'] = pd.DataFrame(v_data)
        
        # Extracción de Costos Marginales
        cml_data = []
        for (b, t), constr in self.model.power_balance_constr.items():
            cml_data.append({'barra': b, 'paso_horario': t, 'costo_marginal_usd_mwh': self.model.dual[constr]})
        results_dfs['costo_marginal'] = pd.DataFrame(cml_data)

        # Extracción de la Demanda
        demanda_data = []
        for b in self.data.B:
            for t in self.data.T:
                demanda_val = self.data.Demanda.get((b, t), 0)
                if demanda_val > 0:
                    demanda_data.append({'barra': b, 'paso_horario': t, 'demanda_mw': demanda_val})
        results_dfs['demanda'] = pd.DataFrame(demanda_data)

        # Se llama a la función auxiliar (modificada para usar 'paso_horario')
        results_dfs = self._add_marginal_generator_info(results_dfs)

        # --- NUEVO: CONVERSIÓN DE PASO HORARIO A FECHA Y HORA ---
        # Asumimos que la simulación empieza el 1 de enero de un año base (ej. 2024)
        año_base = getattr(self.data, 'Año', 2025)
        start_date = pd.to_datetime(f'{año_base}-01-01')

        # DataFrames que contienen la columna de tiempo a convertir
        dfs_con_tiempo = ['generacion', 'vertimiento', 'flujo', 'costo_marginal', 'demanda', 'nivel_embalse_horario']
        
        for df_name in dfs_con_tiempo:
            if df_name in results_dfs and not results_dfs[df_name].empty:
                df = results_dfs[df_name]
                
                # 1. Crear la columna de fecha y hora directamente desde los timestamps
                # Esta columna será de tipo datetime64[ns], ideal para análisis
                df['fecha_hora'] = start_date + pd.to_timedelta(df['paso_horario'], unit='h')
                
                # 2. Eliminar la columna auxiliar 'paso_horario'
                df.drop(columns=['paso_horario'], inplace=True)
                
                # 3. Reordenar para que la fecha y hora aparezcan primero
                columnas_existentes = [col for col in df.columns if col != 'fecha_hora']
                df = df[['fecha_hora'] + columnas_existentes]
                
                results_dfs[df_name] = df
        
        print("Extracción de resultados completada.")
        return results_dfs
      
    def save_results_to_csv(self, output_folder="../resultados"):
        """
        Guarda los DataFrames de resultados en archivos CSV dentro de una carpeta específica.
        """
        results_data = self.get_results_as_dataframes()
        if not results_data:
            print("No hay resultados para guardar.")
            return

        # Crear la carpeta de resultados si no existe
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nGuardando resultados en la carpeta: '{output_folder}'")

        for name, df in results_data.items():
            filepath = os.path.join(output_folder, f"resultados_{name}.csv")
            df.to_csv(filepath, index=False, decimal='.', sep=';')
            print(f" - Archivo '{filepath}' guardado exitosamente.")

if __name__ == '__main__':
    api = ProjectDataAPI(data_path='data')

    # --- Escenario 1: Hidrología MEDIA ---
    print("\n--- INICIANDO SIMULACIÓN: HIDROLOGÍA MEDIA ---")
    model_data_media = ModelData(data_api=api, hidrologia='media', months_to_run=120)
    opt_model_media = OptimizationModel(model_data=model_data_media)
    opt_model_media.solve_and_get_duals(solver_name='cplex')
    opt_model_media.save_results_to_csv(output_folder="resultados_simulacion_MEDIA")

    # --- Escenario 2: Hidrología SECA (P10) ---
    print("\n--- INICIANDO SIMULACIÓN: HIDROLOGÍA SECA (P10) ---")
    model_data_seca = ModelData(data_api=api, hidrologia='P10', months_to_run=120)
    opt_model_seca = OptimizationModel(model_data=model_data_seca)
    opt_model_seca.solve_and_get_duals(solver_name='cplex')
    opt_model_seca.save_results_to_csv(output_folder="resultados_simulacion_SECA")

    # --- Escenario 3: Hidrología HÚMEDA (P90) ---
    print("\n--- INICIANDO SIMULACIÓN: HIDROLOGÍA HÚMEDA (P90) ---")
    model_data_humeda = ModelData(data_api=api, hidrologia='P90', months_to_run=120)
    opt_model_humeda = OptimizationModel(model_data=model_data_humeda)
    opt_model_humeda.solve_and_get_duals(solver_name='cplex')
    opt_model_humeda.save_results_to_csv(output_folder="resultados_simulacion_HUMEDA")