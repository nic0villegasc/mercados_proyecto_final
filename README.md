# (WIP) Proyecto Final: Mercados Eléctricos 202501 - Proyección de Precios

Este repositorio contiene los datos y la descripción para la primera parte del proyecto final del curso Mercados Eléctricos, centrado en la proyección de precios de la energía en un sistema eléctrico de potencia.

## Descripción del Problema

El objetivo principal es proyectar los precios de la energía para los próximos 10 años en un sistema eléctrico simulado compuesto por 3 barras (nodos). El modelo debe considerar diferentes tipos de centrales de generación, perfiles de demanda con crecimiento vegetativo y restricciones en las líneas de transmisión.

Además, se debe realizar un análisis de sensibilidad del precio de la energía frente a tres hidrologías distintas (seca, media y húmeda), optimizando el uso del recurso hídrico a través del cálculo del "Valor del Agua".

## Tareas Pendientes (To-Do)

El estado actual del proyecto es el siguiente:

  - [x] **Extracción y Estructuración de Datos:** Todos los datos iniciales de centrales, demanda, líneas y costos han sido procesados y organizados en archivos `json` y `csv`.
  - [ ] **Cálculo del Valor del Agua:** Desarrollar e implementar la metodología para la optimización del recurso hídrico de las centrales hidroeléctricas a lo largo del horizonte de planificación.
  - [ ] **Desarrollo del Modelo de Optimización:** Implementar el modelo de despacho económico que minimice los costos de operación del sistema, sujeto a todas las restricciones técnicas (mínimos técnicos, capacidad de líneas, etc.).
  - [ ] **Simulación y Proyección a 10 años:** Ejecutar el modelo para los 3 escenarios hidrológicos (seco, medio, húmedo) para obtener las proyecciones.
  - [ ] **Generación de Resultados:** Escribir los scripts necesarios para exportar los entregables solicitados (costos marginales, generación por central, cotas de embalses) en un formato claro.
  - [ ] **Análisis de Resultados:** Redactar las conclusiones del estudio, comparando los efectos de las distintas hidrologías en los precios y la operación del sistema.


## Descripción del Sistema Eléctrico

El sistema está formado por 3 barras interconectadas por 2 líneas de transmisión:

  * **Barra 1 ("Norte"):**
      * Consumo industrial.
      * Central Solar.
      * Central Térmica.
  * **Barra 2 ("Centro"):**
      * Consumo residencial.
      * Central Solar.
      * Central Térmica.
      * Central Hidroeléctrica.
  * **Barra 3 ("Sur"):**
      * Consumo residencial.
      * Central Diésel.
      * Central Hidroeléctrica.

### Líneas de Transmisión

1.  **Línea 1-2:** Conecta la barra Norte y Centro.
2.  **Línea 2-3:** Conecta la barra Centro y Sur.

Ambas líneas tienen una capacidad máxima de transmisión que no puede ser excedida.

## Características de los Componentes

### Centrales de Generación

  * **Solares:** Tienen una generación horaria fija para cada mes (determinística) y su costo variable es de 0 USD/MWh. Pueden verter el exceso de energía.
  * **Térmicas:** Poseen un mínimo técnico de generación. Su operación es mensual (o generan durante todo el mes o no generan). El costo variable depende del precio del combustible.
  * **Diésel:** No tiene mínimo técnico y su costo variable también depende del precio del combustible.
  * **Hidroeléctricas:** Su operación debe ser optimizada. Se debe proponer un método para calcular el "Valor del Agua" para gestionar el recurso hídrico a lo largo del tiempo. La proyección se realiza para 3 escenarios de afluentes (seco, medio, húmedo).

### Demanda

  * Se cuenta con un perfil de consumo horario y mensual definido.
  * Se debe incorporar un crecimiento vegetativo anual sobre la demanda base.

## Estructura del Repositorio

  * `Proyecto Final.pdf`: Documento con la declaración oficial del proyecto, sus requisitos y objetivos.
  * `/data`: Directorio que contiene todos los datos de entrada necesarios para el modelo.
      * `centrales.json`: Características principales y estáticas de las centrales.
      * `lineas_transmision.json`: Capacidad de las líneas de transmisión.
      * `demanda[1-3].csv`: Perfiles de demanda para cada barra.
      * `/datos_centrales`: Contiene los datos variables y específicos por tipo de central, como costos, caudales, perfiles de generación solar, etc.

## Entregables Esperados

El modelo debe generar los siguientes resultados para cada hora de cada mes durante los próximos 10 años:

1.  **Proyección del Costo Marginal ($/MWh)** en cada barra.
2.  **Generación de cada central (MWh)**.
3.  **Nivel de los embalses (cotas)** de las centrales hidroeléctricas.
