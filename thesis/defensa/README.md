# Defensa del TFG

Presentación Beamer en español y formato 16:9. El diseño adapta a LaTeX las composiciones del apartado 6.1 del Manual de Identidad Corporativa de la Universidad de Oviedo. No es una plantilla Beamer oficialmente publicada o aprobada por la Universidad.

## Entrega

- `Defensa_TFG.pdf`: 39 diapositivas para exponer y dos páginas de referencias.
- `Defensa_TFG.pptx`: las mismas 41 páginas como imágenes de 1920 × 1080, con notas del presentador. Los textos y gráficos se editan en los fuentes Beamer. Los enlaces de navegación se conservan en el PDF.
- `Guion_defensa.pdf`: tiempos, explicación por diapositiva, transiciones y preparación de preguntas.
- `inventario_figuras.json`: relación de las 22 figuras científicas con las páginas de la presentación.
- `data/numeric_provenance.json`: trazabilidad de cada cifra destacada hasta su agregado canónico.
- `Fuentes_defensa_TFG.zip`: proyecto autónomo con fuentes, figuras, datos agregados utilizados y PDF.

Los tiempos son objetivos de ensayo. La copia pública localizada de las directrices de la EPI menciona 15–20 minutos orientativos y deja el tiempo al tribunal. El procedimiento vigente de 2026–2027 está enlazado en SharePoint y requiere acceso institucional. Se debe confirmar la duración concreta con el tribunal.

## Orden de la exposición

La introducción explica por qué analizar el trazado cuando el ECG se conserva como imagen. Después del estado del arte se presenta la anatomía del latido, seguida del caso de Brugada en V1. Tras las implementaciones, la tabla de escenarios enlaza con los tres bloques de resultados y precede al protocolo LOOCV. Cada bloque abre con una diapositiva que identifica el origen de los ejemplos y los datos de prueba.

| Diapositivas | Bloque |
| --- | --- |
| 1–6 | Portada, motivación cardiovascular, ECG como imagen, propuesta, índice y estado del arte |
| 7–11 | Lectura del latido, Brugada en V1, CNN, aprendizaje en contexto y adaptación de dominio |
| 12–16 | Datos, simulador, familias, etiquetas y conjunto clínico |
| 17–22 | Diseño experimental, implementaciones, escenarios y protocolo |
| 23–34 | Resultados en el simulador, cambio de dominio y evaluación con ejemplos clínicos |
| 35–39 | Ética, discusión, conclusiones, continuación y cierre |

Exactitud balanceada y F1 comparten diapositiva en los escenarios de simulación y aprendizaje con ejemplos clínicos. El guion suma 18 minutos.

## Editar y compilar

El archivo `main.tex` define el orden de las diapositivas. `content.tex` contiene las diapositivas y sus notas, y `tema-uniovi.sty` define el diseño. `guion.tex` es el documento de ensayo.

Con Tectonic instalado, ejecutar desde esta carpeta:

```powershell
python build.py --compile
```

El script también admite `--compiler RUTA_AL_TECTONIC` y detecta `latexmk` con XeLaTeX. Se utiliza Georgia cuando está instalada; TeX Gyre Termes es la alternativa libre. El ZIP no redistribuye tipografías de Microsoft. La primera compilación con Tectonic puede requerir descargar paquetes de LaTeX.

`slides.json` conserva el contenido estructurado, los tiempos y las notas. Si se edita, `python build.py --generate` vuelve a generar los fuentes TeX; esta opción sobrescribe las ediciones directas de esos fuentes. La compilación normal no los regenera.

`python build.py --package` vuelve a preparar el ZIP. Las compilaciones intermedias se guardan en `.build/`.

Para generar el PowerPoint desde el PDF ya compilado, ejecutar `node export_powerpoint.mjs` con Poppler (`pdftoppm`) y el módulo `@oai/artifact-tool` del entorno de Codex. Si el módulo no está disponible en `node_modules`, `ARTIFACT_TOOL_MODULE` puede apuntar a su archivo `dist/artifact_tool.mjs`. `PDFTOPPM` permite indicar la ruta del ejecutable. El script acepta una ruta de salida opcional.

La compilación y el empaquetado usan la biblioteca estándar de Python. La comprobación opcional `python validate.py` requiere `pypdf` y `pypdfium2`. Verifica las páginas, los enlaces, la cobertura de las figuras inventariadas y la trazabilidad de las cifras. `--refresh` requiere además `pypdf` para generar los recortes vectoriales.

`--refresh` solo debe utilizarse desde el repositorio original: actualiza las copias de los agregados canónicos, la procedencia numérica y las figuras de presentación. No ejecuta experimentos ni escribe en la campaña final. El proyecto ya incluye todo lo necesario para compilar sin acceder al repositorio original.

## Figuras y fuentes

Los PDF se incluyen directamente, conservando sus trazados y texto vectoriales. Los gráficos multipanel se amplían mediante recortes PDF, sin rasterización ni alteración de sus series. El inventario documenta las figuras originales y sus recortes. El esquema general de ICL de Dong contiene imágenes ráster internas y se conserva como tal.

Los ejemplos experimentales de ECG proceden de la memoria. La introducción, el estado del arte, los fundamentos de CNN y la presentación del HUCA incorporan [recursos visuales publicados con atribución](assets/referencias_visuales/README.md). Las métricas se presentan de forma agregada y el paquete no contiene identificadores ni predicciones individuales de pacientes del estudio.

Las dos diapositivas de referencias agrupan las fuentes de contexto clínico, datos, antecedentes, métodos y métricas que se utilizan en la defensa. Las licencias y los créditos visuales se documentan por separado.

## Procedencia de las etiquetas del HUCA

La sección «Datos y tareas» de la memoria explica que se excluyeron 46 de los 363 casos por recomendación de los médicos colaboradores, ante dudas de clasificación. En la copia de trabajo, el constructor excluyó los registros con `brugada=1` y `basal_pattern=1`. La auditoría del repositorio confirmó que este filtro reproduce los 317 pacientes y las 116 etiquetas positivas de la campaña clínica canónica. Las etiquetas de la copia de trabajo y las de los metadatos públicos difieren. La ficha pública describe la resolución de casos ambiguos por consenso para su propia publicación. [Brugada-HUCA en PhysioNet](https://physionet.org/content/brugada-huca/1.0.0/)

Referencias de diseño y preparación:

- [Identidad corporativa de UniOvi](https://www.uniovi.es/conocenos/uniovi/identidad), manual, apartado 6.1, páginas 57–58.
- [TFG de la EPI](https://epigijon.uniovi.es/estudiantes/tfg).
- [Copia pública de las directrices de la EPI](https://expydoc.com/doc/6397600/directrices-para-los-estudiantes), referencia orientativa de duración.
- [MIT Communication Lab: Slide Design](https://mitcommlab.mit.edu/aeroastro/commkit/slide-design/).
- [Beamer User Guide](https://ctan.org/pkg/beamer), capítulo 5.
