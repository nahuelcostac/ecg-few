# Espejo agregado de los resultados finales

Esta carpeta contiene el espejo versionado de `outputs/final_results_tfg/loocv/`. Incluye el paquete final `Final Results TFG`, importado el 30 de agosto de 2026, y la campaña clínica binaria promovida de forma aditiva el 25 de octubre de 2026.

La fuente canónica completa es `outputs/final_results_tfg/`. Las huellas de sus dos paquetes están registradas en `SOURCE.sha256` y `SOURCE_CNN_CLINICAL.sha256`. Si este espejo y la fuente canónica difieren, prevalece la fuente canónica y este directorio debe volver a sincronizarse.

## Uso

- Utilizar los CSV y JSON agregados de esta carpeta para auditar cifras y regenerar gráficas.
- No mezclar estos datos con campañas anteriores, resultados temporales ni ejecuciones nuevas.
- Tratar las imágenes y el texto de la memoria como derivados, no como fuentes numéricas.
- Mantener los experimentos futuros fuera de esta carpeta hasta que el usuario autorice expresamente una nueva campaña final.

Las salidas completas por ejecución no se versionan aquí. Se conservan localmente en `outputs/final_results_tfg/runs/` y pueden contener información granular; no deben usarse para exponer datos a nivel de paciente.
