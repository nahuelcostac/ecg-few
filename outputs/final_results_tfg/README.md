# Resultados finales canónicos del TFG

Esta es la única ubicación autoritativa para los resultados experimentales del proyecto. El contenido procede de `Final Results TFG`, recibido e importado el 30 de agosto de 2026. No debe combinarse con campañas anteriores.

## Estructura

- `source/final_results_tfg.zip`: archivo original, conservado sin cambios.
- `loocv/`: informes agregados, métricas, curvas y matrices de confusión finales.
- `runs/`: salidas completas de las ejecuciones finales.
- `tfg_figures/`: selección de figuras finales y su manifiesto.
- `SOURCE.sha256`: huella del ZIP original.

`reports/loocv/` es el espejo versionado de `loocv/`. Las figuras presentes en `tfg_figures/results/` se copian a `thesis/thesis/assets/results/` cuando forman parte de la memoria. Esas ubicaciones derivadas no constituyen fuentes independientes.

El ZIP y las salidas completas permanecen fuera de Git por tamaño y porque pueden contener información granular. No se deben exponer datos a nivel de paciente. Para análisis compartibles deben usarse los informes agregados.

Los nuevos experimentos deben escribirse en otra carpeta, por ejemplo `outputs/work/<nombre_de_campaña>/`, y considerarse provisionales hasta que el usuario autorice expresamente reemplazar esta campaña.
