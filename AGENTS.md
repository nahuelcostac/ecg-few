# Instrucciones para agentes de IA

Estas instrucciones se aplican a todo el repositorio.

## Fuente única de resultados finales

- La única fuente canónica de resultados experimentales es `outputs/final_results_tfg/`.
- El paquete base fue importado de `outputs/final_results_tfg/source/final_results_tfg.zip`. Su SHA-256 está registrado en `outputs/final_results_tfg/SOURCE.sha256`.
- La campaña binaria clínica se promovió de forma aditiva desde `outputs/work/cnn_clinical_brugada_loocv/`. Su archivo, hash y procedencia están en `source/cnn_clinical_brugada_loocv.zip`, `SOURCE_CNN_CLINICAL.sha256` y `CNN_CLINICAL_PROMOTION.json`.
- `outputs/final_results_tfg/loocv/` contiene los informes agregados finales; `outputs/final_results_tfg/runs/` contiene las salidas completas por ejecución; y `outputs/final_results_tfg/tfg_figures/` contiene la selección final de figuras.
- `reports/loocv/` es únicamente el espejo versionado de los informes agregados finales. Si alguna vez difiere de la carpeta canónica, prevalece `outputs/final_results_tfg/loocv/` y debe corregirse el espejo.
- Las figuras de `thesis/thesis/assets/results/`, el texto de la memoria y las tablas del `README.md` son artefactos de presentación derivados. No deben utilizarse como fuente numérica primaria.

No se deben consultar, mezclar, restaurar ni citar resultados de ejecuciones anteriores, carpetas temporales, copias de seguridad o cualquier otro directorio de salida. Tampoco se debe elegir una campaña por la fecha de modificación de sus archivos.

## Trabajo con nuevos experimentos

- No sobrescribir, regenerar ni modificar `outputs/final_results_tfg/` salvo petición expresa del usuario de sustituir la campaña final.
- Guardar cualquier experimento nuevo en una ubicación separada, por ejemplo `outputs/work/<nombre_de_campaña>/`, y etiquetarlo como provisional.
- No promover un experimento nuevo a resultado final ni sincronizarlo con `reports/loocv/` sin autorización expresa.
- Al actualizar cifras en documentación o en la memoria, rastrear cada valor hasta un CSV o JSON de la fuente canónica y mantener sincronizados los artefactos derivados.
- Las salidas completas pueden contener identificadores o información granular. No exponer datos a nivel de paciente; para documentación y análisis compartibles, preferir siempre los agregados de `reports/loocv/`.

## Conservación

No eliminar ninguno de los ZIP fuente ni sus archivos SHA-256. Si el usuario autoriza una campaña final nueva, registrar su procedencia y hash, actualizar de forma atómica la ubicación canónica, sincronizar el espejo agregado y revisar esta guía y los README relacionados.
