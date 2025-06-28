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
    def __init__(self, data_api: ProjectDataAPI, months_to_run: int = None):
        """
        Inicializa y construye los conjuntos a partir de la API de datos.
        """
        self.api = data_api
        self.months_to_run = months_to_run
        self._create_all_sets()
        print("Paso 1 completado: Todos los conjuntos e índices han sido creados usando la API.")
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
                
      # --- Parámetro: Valor del Agua (Costo de Oportunidad) ---
      self.ValorAgua = {}
      self.NivelEmbalseMensual = {} # Para guardar los resultados de la simulación

      centrales_df = self.api.get_centrales_info()
      costos_termicos_df = pd.DataFrame(
          [(g, t, c) for (g, t), c in self.CostoVar.items()],
          columns=['generador', 'hora', 'costo']
      )

      for g in self.G_hidro:
          # 1. Obtener datos específicos de la central hidroeléctrica
          info_hidro = centrales_df[centrales_df['id_central'] == g].iloc[0]
          cota_inicial = info_hidro['cota_inicial_mwh']
          cota_maxima = info_hidro['cota_maxima_mwh']
          
          # Usaremos hidrología 'media' como base para el cálculo del valor del agua
          # Puedes cambiar 'media' a 'seca_p10' o 'humeda_p90' para otros escenarios
          caudal_df = self.api.get_caudal(g, hidrologia='media')
          caudal_map = caudal_df.set_index(['Año', 'Mes'])['Caudal_MWh'].to_dict()

          nivel_actual_mwh = cota_inicial

          # 2. Simular mes a mes para calcular el costo de oportunidad
          for m in self.M:
            horas_del_mes = self.T_m[m]
            timestamp_mes = self.hourly_mapping_num_to_ts[horas_del_mes[0]]
            año_relativo = timestamp_mes.year - 2025 + 1
            mes_calendario = timestamp_mes.month
            
            # 3. Calcular el porcentaje de llenado del embalse
            caudal_afluente = caudal_map.get((año_relativo, mes_calendario), 0)
            agua_disponible = nivel_actual_mwh + caudal_afluente
            porcentaje_llenado = min(agua_disponible / cota_maxima, 1.0) # Cap at 100%

            # 4. Encontrar costos térmicos de referencia para este mes
            costos_del_mes = costos_termicos_df[costos_termicos_df['hora'].isin(horas_del_mes)]
            costo_termico_min = costos_del_mes['costo'].min()
            costo_termico_max = costos_del_mes['costo'].max()
            
            # 5. Aplicar la heurística de interpolación lineal
            costo_oportunidad_bajo = 0.20 * costo_termico_min
            costo_oportunidad_alto = 1.20 * costo_termico_max
            
            # Interpolación lineal invertida: más lleno = más barato
            valor_agua_mes = costo_oportunidad_alto - (porcentaje_llenado * (costo_oportunidad_alto - costo_oportunidad_bajo))
            
            # 6. Asignar el valor a todas las horas del mes
            for t in horas_del_mes:
                self.ValorAgua[(g, t)] = valor_agua_mes
            
            # 7. Actualizar el nivel del embalse para el siguiente mes
            # Asumimos que se usa el caudal afluente (operación a filo de agua) para mantener el nivel
            # El nivel solo cambia por la diferencia entre la cota y el agua usada.
            agua_turbinada = caudal_afluente
            nivel_actual_mwh = min(nivel_actual_mwh + caudal_afluente - agua_turbinada, cota_maxima)
            
            # Guardamos el resultado para análisis posterior
            self.NivelEmbalseMensual[(g, m)] = nivel_actual_mwh
          
      print(f"Parámetro 'ValorAgua' creado con {len(self.ValorAgua)} entradas.")
      
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
              # 1. Obtenemos el DataFrame del perfil de generación para el generador solar 'g'
              solar_df = self.api.get_time_series_data(g, 'generacion_fija')
              
              # 2. Creamos un mapeo eficiente de (Año, Mes, Hora) -> Generacion_MW
              # Esto es mucho más rápido que buscar en el DataFrame miles de veces.
              solar_map = solar_df.set_index(['Año', 'Mes', 'Hora'])['Generacion_MW'].to_dict()

              # 3. Iteramos sobre TODAS las horas 't' del horizonte de simulación
              for t in self.T:
                  # Obtenemos el timestamp (fecha y hora exactas) para la hora numérica 't'
                  timestamp = self.hourly_mapping_num_to_ts[t]
                  
                  # Calculamos el año relativo, mes y hora para ese timestamp
                  año_relativo = timestamp.year - 2025 + 1
                  mes = timestamp.month
                  hora_del_dia = timestamp.hour # La hora de 0 a 23
                  
                  # Buscamos la generación solar en nuestro mapeo
                  generacion_hora = solar_map.get((año_relativo, mes, hora_del_dia))
                  
                  if generacion_hora is not None:
                      # Creamos la entrada en el diccionario para Pyomo: (generador, hora) -> generacion
                      self.PdispSolar[(g, t)] = generacion_hora
                  else:
                      # Si no se encuentra, asumimos 0 (por ejemplo, para datos faltantes)
                      self.PdispSolar[(g, t)] = 0

          except (ValueError, KeyError) as e:
              print(f"Advertencia al buscar 'generacion_fija' para {g}: {e}")

      print(f"Parámetro 'PdispSolar' creado con {len(self.PdispSolar)} entradas.")

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

    def _build_objective(self):
        """
        Define la función objetivo del modelo: minimizar el costo total de operación.
        """
        print("Construyendo la función objetivo...")
        
        # Generadores con costo variable (combustible)
        generadores_costo_combustible = self.data.G_termica + self.data.G_diesel
        
        # Costo total por combustible
        costo_combustible = sum(self.data.CostoVar[g, t] * self.model.p[g, t] 
                                for g in generadores_costo_combustible 
                                for t in self.data.T 
                                if (g, t) in self.data.CostoVar)
        
        # Costo total por uso de agua (costo de oportunidad)
        costo_agua = sum(self.data.ValorAgua[g, t] * self.model.p[g, t] 
                         for g in self.data.G_hidro 
                         for t in self.data.T 
                         if (g, t) in self.data.ValorAgua)
        
        # La función objetivo es la suma de ambos costos
        costo_total = costo_combustible + costo_agua
        
        self.model.objective = pyo.Objective(expr=costo_total, sense=pyo.minimize)
        
        print("Función objetivo construida exitosamente.")
        
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
    
    def get_results_as_dataframes(self):
        """
        Extrae los resultados de las variables del modelo resuelto y los
        convierte en DataFrames de pandas para un fácil análisis.
        
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
            if pyo.value(var, exception=False) > 1e-6: # Solo guardar si hay generación
                p_data.append({'generador': g, 'hora': t, 'generacion_mw': pyo.value(var)})
        results_dfs['generacion'] = pd.DataFrame(p_data)
        
        # Estado de encendido mensual (u)
        u_data = []
        for (g, m), var in self.model.u.items():
            u_data.append({'generador': g, 'mes': m, 'encendido': int(pyo.value(var))})
        results_dfs['compromiso_unidad'] = pd.DataFrame(u_data)

        # Vertimiento solar (p_vert)
        vert_data = []
        for (g, t), var in self.model.p_vert.items():
            if pyo.value(var, exception=False) > 1e-6: # Solo guardar si hay vertimiento
                vert_data.append({'generador': g, 'hora': t, 'vertimiento_mw': pyo.value(var)})
        results_dfs['vertimiento'] = pd.DataFrame(vert_data)

        # Flujo en líneas (f)
        f_data = []
        for (l, t), var in self.model.f.items():
            flow_val = pyo.value(var, exception=False)
            if abs(flow_val) > 1e-6: # Solo guardar si hay flujo
                f_data.append({'linea': l, 'hora': t, 'flujo_mw': flow_val})
        results_dfs['flujo'] = pd.DataFrame(f_data)

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

  model_data = ModelData(data_api=api, months_to_run=120)

  print("\n--- CONJUNTOS DE GENERADORES ---")
  print(f"G (Todos): {model_data.G}")
  print(f"G_termica: {model_data.G_termica}")
  
  print("\n--- CONJUNTOS DE LA RED ---")
  print(f"B (Barras): {model_data.B}")
  print(f"L (Líneas): {model_data.L}")

  print("\n--- CONJUNTOS DE TIEMPO ---")
  print(f"M (Meses): {len(model_data.M)} meses, ej: {model_data.M[:5]}...")
  print(f"T (Horas): {len(model_data.T)} horas en total")
  print(f"T_m (Horas en Mes 1): {len(model_data.T_m.get(1, []))} horas")

  print("\n--- CONJUNTOS DE CONECTIVIDAD ---")
  print(f"G_b (Generadores por Barra): {model_data.G_b}")
  print(f"L_out (Líneas que salen de Barra): {model_data.L_out}")
  print(f"L_in (Líneas que entran a Barra): {model_data.L_in}")  
  
  print("\n--- VALIDACIÓN DE PARÁMETROS ---")
        
  # Seleccionamos un generador y horas específicas para verificar
  generador_a_verificar = 'TERMICA-NORTE-01'
  hora_mes_1 = 1
  
  # La primera hora del mes 2 (asumiendo que el mes 1 tiene 744 horas)
  primera_hora_mes_2 = model_data.T_m.get(1, [])[-1] + 1 

  print(f"Verificando costos para el generador: {generador_a_verificar}")

  costo_h1 = model_data.CostoVar.get((generador_a_verificar, hora_mes_1))
  print(f"  - Costo en la hora {hora_mes_1} (Mes 1): {costo_h1}")

  costo_h2 = model_data.CostoVar.get((generador_a_verificar, primera_hora_mes_2))
  print(f"  - Costo en la hora {primera_hora_mes_2} (Mes 2): {costo_h2}")
  
  print("\n--- VALIDACIÓN PARÁMETRO ValorAgua ---")
  #for generador_hidro in model_data.G_hidro:
  #  plot_hydro_validation(model_data, generador_hidro)
  
  print("\n--- VALIDACIÓN PARÁMETROS Pmin, Pmax_inst y EnergiaMaxMensual ---")

  gen_termica = model_data.G_termica[0] if model_data.G_termica else None
  gen_hidro = model_data.G_diesel[0] if model_data.G_hidro else None

  if gen_termica:
      pmin_termica = model_data.Pmin.get(gen_termica)
      pmax_inst_termica = model_data.Pmax_inst.get(gen_termica)
      energia_max_mes_1 = model_data.EnergiaMaxMensual.get((gen_termica, 1))
      energia_max_mes_2 = model_data.EnergiaMaxMensual.get((gen_termica, 2))
      
      print(f"Verificando {gen_termica} (Térmica):")
      print(f"  - Pmin: {pmin_termica} MW")
      print(f"  - Pmax Instantáneo: {pmax_inst_termica} MW (constante)")
      print(f"  - Energía Máx Mensual (Mes 1): {energia_max_mes_1} MWh")
      print(f"  - Energía Máx Mensual (Mes 2): {energia_max_mes_2} MWh (debería ser diferente)")

  if gen_hidro:
      pmin_hidro = model_data.Pmin.get(gen_hidro)
      pmax_inst_hidro = model_data.Pmax_inst.get(gen_hidro)
      energia_max_hidro_mes_1 = model_data.EnergiaMaxMensual.get((gen_hidro, 1))
      
      print(f"Verificando {gen_hidro} (Hidro):")
      print(f"  - Pmin: {pmin_hidro} MW")
      print(f"  - Pmax Instantáneo: {pmax_inst_hidro} MW (constante)")
      print(f"  - Energía Máx Mensual (Mes 1): {energia_max_hidro_mes_1} MWh")
      
  print("\n--- VALIDACIÓN PARÁMETRO Demanda ---")
  # Verificamos la demanda para cada barra en la primera hora
  hora_a_verificar = 1
  for barra in model_data.B:
      demanda = model_data.Demanda.get((barra, hora_a_verificar))
      print(f"  - Demanda en Barra '{barra}' en Hora {hora_a_verificar}: {demanda} MW")  
      
  print("\n--- VALIDACIÓN PARÁMETRO FlujoMax ---")
  if model_data.FlujoMax:
      for linea, capacidad in model_data.FlujoMax.items():
          print(f"  - Capacidad Máxima de la línea '{linea}': {capacidad} MW")
  else:
      print("  - No se cargaron datos de capacidad de líneas.")
    
  print("\n--- VALIDACIÓN PARÁMETRO PdispSolar ---")
  
  if model_data.G_solar:
      gen_solar_a_verificar = model_data.G_solar[0]
      
      # Verificamos la primera hora del día (debería ser 0)
      hora_noche = 1 
      # Verificamos una hora de mediodía (debería ser > 0)
      hora_dia = 13 

      gen_noche = model_data.PdispSolar.get((gen_solar_a_verificar, hora_noche))
      gen_dia = model_data.PdispSolar.get((gen_solar_a_verificar, hora_dia))

      print(f"Verificando generación disponible para: {gen_solar_a_verificar}")
      print(f"  - Generación Solar en Hora {hora_noche} (noche): {gen_noche} MW")
      print(f"  - Generación Solar en Hora {hora_dia} (mediodía): {gen_dia} MW")
  else:
      print("  - No hay generadores solares para validar.")
      
  # 2. Construir el modelo de optimización
  opt_model = OptimizationModel(model_data=model_data)

  # 3. Validar la creación de la variable
  print("\n--- VALIDACIÓN PASO 3: VARIABLES ---")
  
  # Verificamos que el componente 'p' exista en el modelo
  if hasattr(opt_model.model, 'p'):
      print("\nVariable 'p' (potencia despachada) ha sido creada exitosamente.")
      
      # Contamos cuántas variables individuales se crearon para 'p'
      # Debería ser |G| * |T|
      num_variables_p = len(opt_model.model.p)
      print(f"  - Número total de variables 'p' individuales: {num_variables_p}")
      print(f"  - (Calculado como |G|={len(model_data.G)} * |T|={len(model_data.T)} = {len(model_data.G) * len(model_data.T)})")
      
      # Mostramos un ejemplo de una instancia de variable (aún sin valor)
      primer_gen = model_data.G[0]
      primera_hora = model_data.T[0]
      print(f"  - Ejemplo de una instancia de variable: model.p['{primer_gen}', {primera_hora}]")

  else:
      print("ERROR: La variable 'p' no fue encontrada en el modelo.")
      
  # Verificamos la variable 'u'
  if hasattr(opt_model.model, 'u'):
      print("\nVariable 'u' (estado de encendido) ha sido creada exitosamente.")
      
      # Contamos cuántas variables individuales se crearon para 'u'
      # Debería ser |G_termica + G_diesel| * |M|
      num_variables_u = len(opt_model.model.u)
      generadores_uc = model_data.G_termica + model_data.G_diesel
      print(f"  - Número total de variables 'u' individuales: {num_variables_u}")
      print(f"  - (Calculado como |G_uc|={len(generadores_uc)} * |M|={len(model_data.M)} = {len(generadores_uc) * len(model_data.M)})")
      
      # Mostramos un ejemplo de una instancia de variable (aún sin valor)
      primer_gen_uc = generadores_uc[0]
      primer_mes = model_data.M[0]
      print(f"  - Ejemplo de una instancia de variable: model.u['{primer_gen_uc}', {primer_mes}]")
  else:
      print("ERROR: La variable 'u' no fue encontrada en el modelo.")
      
  # Verificamos la nueva variable 'p_vert'
  if hasattr(opt_model.model, 'p_vert'):
      print("\nVariable 'p_vert' (vertimiento solar) ha sido creada exitosamente.")
      
      # Contamos cuántas variables individuales se crearon para 'p_vert'
      num_variables_p_vert = len(opt_model.model.p_vert)
      print(f"  - Número total de variables 'p_vert': {num_variables_p_vert}")
      print(f"  - (Calculado como |G_solar|={len(model_data.G_solar)} * |T|={len(model_data.T)} = {len(model_data.G_solar) * len(model_data.T)})")

  else:
      print("ERROR: La variable 'p_vert' no fue encontrada en el modelo.")
      
  # Verificamos la nueva variable 'f'
  if hasattr(opt_model.model, 'f'):
      print("\nVariable 'f' (flujo de potencia) ha sido creada exitosamente.")
      
      # Contamos cuántas variables individuales se crearon para 'f'
      num_variables_f = len(opt_model.model.f)
      print(f"  - Número total de variables 'f': {num_variables_f}")
      print(f"  - (Calculado como |L|={len(model_data.L)} * |T|={len(model_data.T)} = {len(model_data.L) * len(model_data.T)})")

  else:
      print("ERROR: La variable 'f' no fue encontrada en el modelo.")
      
  # Verificamos que la función objetivo exista
  if hasattr(opt_model.model, 'objective'):
      print("OK: Función Objetivo 'objective' creada exitosamente.")
      print(f"  - Sentido de la optimización: {'Minimizar' if opt_model.model.objective.sense == pyo.minimize else 'Maximizar'}")
  else:
      print("ERROR: La Función Objetivo no fue encontrada en el modelo.")
  
  print("\nModelo construido exitosamente. Listo para el Paso 5: Definir Restricciones.")
  
  # Verificamos que la restricción de balance de potencia exista
  if hasattr(opt_model.model, 'power_balance_constr'):
      print("OK: Restricción 'power_balance_constr' creada exitosamente.")
  else:
      print("ERROR: La restricción 'power_balance_constr' no fue encontrada.")
  
  # Verificamos que la restricción de capacidad de línea exista
  if hasattr(opt_model.model, 'line_capacity_constr'):
      print("OK: Restricción 'line_capacity_constr' creada exitosamente.")
  else:
      print("ERROR: La restricción 'line_capacity_constr' no fue encontrada.")
  
  # Verificamos que la nueva restricción de límites térmicos exista
  if hasattr(opt_model.model, 'thermal_limits_constr'):
      print("OK: Restricción 'thermal_limits_constr' creada exitosamente.")
  else:
      print("ERROR: La restricción 'thermal_limits_constr' no fue encontrada.")
      
  # Verificamos que la nueva restricción de balance solar exista
  if hasattr(opt_model.model, 'solar_balance_constr'):
      print("OK: Restricción 'Balance de Generación Solar' creada.")
  else:
      print("ERROR: La restricción 'solar_balance_constr' no fue encontrada.")
  
  # Verificamos la nueva restricción para centrales flexibles
  if hasattr(opt_model.model, 'flexible_generation_limits_constr'):
      print("OK: Restricción 'Límites de Generación Flexible' creada.")
  else:
      print("ERROR: La restricción 'flexible_generation_limits_constr' no fue encontrada.")
  
  # Verificamos la nueva restricción para energia mensual
  if hasattr(opt_model.model, 'monthly_energy_budget_constr'):
    print("OK: Restricción 'Presupuesto Mensual de Energía' creada.")
  else:
      print("ERROR: La restricción 'Presupuesto Mensual de Energía' no fue encontrada.")
  
  print("\n¡FORMULACIÓN COMPLETA! El modelo está listo para ser resuelto.")
  
  # 3. Validar la creación del modelo (opcional, se puede comentar)
  print("\n--- VALIDACIÓN DEL MODELO CONSTRUIDO ---")
  print(f"Número de variables: {opt_model.model.nvariables()}")
  print(f"Número de restricciones: {opt_model.model.nconstraints()}")
  
  #opt_model.export_model_to_file("modelo_para_cplex")
  
  #debug_infeasibility_v3(model_data=model_data)
  
  # 4. Resolver el modelo
  # Asegúrate de tener CPLEX instalado y accesible en el PATH de tu sistema.
  opt_model.solve(solver_name='cplex')

  # 5. Validar los resultados (si la solución fue óptima)
  if opt_model.results.solver.termination_condition == pyo.TerminationCondition.optimal:
    # 1. Obtener el costo total
    costo_total = pyo.value(opt_model.model.objective)
    print(f"\n--- RESULTADOS DE LA OPTIMIZACIÓN ---")
    print(f"Costo Total de Operación para 1 Mes: ${costo_total:,.2f}")
    
    # 2. Obtener los resultados en formato de tablas (DataFrames)
    resultados_tablas = opt_model.get_results_as_dataframes()
    
    if resultados_tablas:
        # 3. Mostrar un resumen de cada tabla de resultados
        print("\n--- Resumen de Compromiso de Unidades (Térmicas) ---")
        print(resultados_tablas['compromiso_unidad'])
        
        print("\n--- Resumen de Generación Horaria (primeras 5 filas con generación) ---")
        print(resultados_tablas['generacion'])
        
        print("\n--- Resumen de Flujo en Líneas (primeras 5 filas con flujo) ---")
        print(resultados_tablas['flujo'].head())

        if not resultados_tablas['vertimiento'].empty:
            print("\n--- Resumen de Vertimiento Solar (primeras 5 filas con vertimiento) ---")
            print(resultados_tablas['vertimiento'].head())
        else:
            print("\nNo hubo vertimiento solar en la solución.")

    opt_model.save_results_to_csv(output_folder="resultados_simulacion")