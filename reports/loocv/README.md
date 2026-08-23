# Resultados agregados utilizados por el TFG

Esta carpeta contiene únicamente los resultados agregados necesarios para regenerar las gráficas y auditar las cifras de la memoria. Proceden de la campaña final cerrada el 12 de agosto de 2026.

## Contenido incluido

- Resúmenes de la CNN por valor de `K` para el simulador y la cohorte del HUCA.
- Resúmenes de adaptación de dominio y comparaciones sintético-real.
- Resúmenes de Gemma 4 por valor de `K`, condición y tarea.
- Metadatos agregados de validación de la configuración VLM.

Los CSV contienen medias, desviaciones estándar, métricas y sumas agregadas de las matrices de confusión. No contienen identificadores de pacientes ni referencias a imágenes individuales.

## Contenido excluido

- ECG e imágenes clínicas.
- Metadatos o manifiestos de pacientes.
- Planes de pliegues LOOCV.
- Predicciones individuales y respuestas del VLM.
- Rutas de ejecución, checkpoints y salidas por semilla.

Estas exclusiones no impiden regenerar las figuras finales, porque los renderizadores consumen directamente los resúmenes agregados incluidos aquí.

## Regeneración y auditoría

Desde la raíz del repositorio:

```bash
RUN_COMPARISONS=0 sh scripts/run/build_tfg_report_figures.sh
uv run --no-sync python scripts/thesis/audit_final_tfg.py
```

`RUN_COMPARISONS=0` evita intentar reconstruir comparaciones desde predicciones individuales, que se excluyen deliberadamente. Las tablas comparativas agregadas necesarias ya están versionadas.
