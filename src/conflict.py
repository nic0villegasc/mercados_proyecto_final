import docplex.mp.model_reader as ModelReader
import docplex.mp.conflict_refiner as cr
from docplex.mp.utils import DOcplexException
from docplex.mp.conflict_refiner import ConstraintsGroup # Importado para el ejemplo avanzado
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# --- Carga del Modelo ---
# Usamos un bloque try-except para manejar el caso de que el archivo no exista.
try:
    # ModelReader.read_model crea y devuelve un nuevo objeto de modelo desde el archivo.
    # Es mejor usar ignore_names=False para ver los nombres originales de tus restricciones.
    model = ModelReader.read_model("model.lp", ignore_names=False)
    print(f"Modelo '{model.name}' cargado exitosamente desde 'model.lp'.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'model.lp'.")
    print("Asegúrate de que el archivo esté en el mismo directorio que este script.")
    exit() # Termina el script si no hay archivo que leer.
except DOcplexException as e:
    print(f"Error de DOcplex al leer el modelo: {e}")
    exit()


# --- Resolución y Análisis del Conflicto ---
print("\nIntentando resolver el modelo...")
model.solve(log_output=False) # log_output=False para una salida más limpia

solve_status = model.get_solve_status()
print(f"Estado de la solución: {solve_status.name}")

# Comprobamos los dos posibles estados de inviabilidad.
infeasible_stati = {"INFEASIBLE_SOLUTION", "INFEASIBLE_OR_UNBOUNDED_SOLUTION"}

if solve_status.name in infeasible_stati:
    print("\n--- Modelo INFACTIBLE. Iniciando el refinador de conflictos. ---")
    cref = cr.ConflictRefiner()
    
    print("\n--- 1. Resumen Automático del Conflicto (display=True) ---")
    cref.refine_conflict(model, display=True)

    print("\n\n--- 2. Análisis Detallado y Programático ---")
    conflict_result = cref.refine_conflict(model, display=False)
    
    if not conflict_result or conflict_result.number_of_conflicts == 0:
        print("El refinador no pudo identificar un conflicto específico.")
    else:
        # La documentación menciona el método display_stats() para un resumen estadístico.
        print("\n--- 2a. Estadísticas del Conflicto (con display_stats()) ---")
        conflict_result.display_stats()

        print("\n--- 2b. Listado Detallado de Elementos en Conflicto ---")
        # CORRECCIÓN: Iteramos directamente sobre los elementos del conflicto.
        # El método `iter_conflicts()` devuelve cada elemento conflictivo uno por uno.
        for conflict_element in conflict_result.iter_conflicts():
            print("-" * 25)
            # `conflict_element` es una tupla con .name, .element, y .status
            print(f"    * Estado      : {conflict_element.status.name}")
            print(f"    * Nombre      : '{conflict_element.name}'")
            print(f"    * Definición  : {conflict_element.element}")
            
            # Identificamos el tipo de elemento para mayor claridad.
            element_type = type(conflict_element.element)
            if "VarLbConstraintWrapper" in str(element_type):
                 print(f"    * Tipo        : Límite Inferior de Variable")
            elif "VarUbConstraintWrapper" in str(element_type):
                 print(f"    * Tipo        : Límite Superior de Variable")
            else:
                 print(f"    * Tipo        : Restricción del Modelo")
        
        # --- NUEVA SECCIÓN: Usando pandas para una mejor visualización ---
        if PANDAS_AVAILABLE:
            print("\n\n--- 3. Visualización del Conflicto como Tabla (DataFrame de Pandas) ---")
            # El código fuente muestra que `as_output_table()` puede generar un DataFrame.
            conflict_df = conflict_result.as_output_table(use_df=True)
            print(conflict_df.to_string())

    print("\n\n--- 4. (Uso Avanzado) Cómo Guiar al Refinador de Conflictos ---")
    print("""
    La documentación que proporcionaste menciona los parámetros 'preferences' y 'groups' 
    para controlar el proceso. Esto es muy útil en modelos grandes para priorizar 
    qué restricciones son más importantes y no deberían ser eliminadas. El ejemplo
    sigue siendo válido para este propósito.
    """)
else:
    print("\nEl modelo se resolvió o el estado no es infactible.")
    if model.solution:
        print("\n--- Solución Encontrada ---")
        model.print_solution()
