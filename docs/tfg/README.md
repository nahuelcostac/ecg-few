# Thesis and defense

This folder publishes the revised Spanish thesis **Análisis de ECGs con grandes modelos de lenguaje: aplicación al síndrome de Brugada**, its defense, and the accompanying LaTeX sources and assets.

| Deliverable | File | Source |
| --- | --- | --- |
| Revised thesis, 81 pages | [Thesis PDF](memoria/TFG_Alejandro_Moris_Lara_FINAL.pdf) | [memoria/main.tex](memoria/main.tex) |
| Defense, 41 slides | [Defense PDF](defensa/Defensa_TFG.pdf) | [defensa/main.tex](defensa/main.tex) |
| PowerPoint with speaker notes | [Defense PPTX](defensa/Defensa_TFG.pptx) | Same Beamer sources and [export script](defensa/export_powerpoint.mjs) |
| Speaker guide | [Guide PDF](defensa/Guion_defensa.pdf) | [defensa/guion.tex](defensa/guion.tex) |

The PowerPoint represents the PDF pages as 3840 × 2160 images, with speaker notes. Edit the Beamer sources to change slide text and figures. The deck contains 38 presentation slides, one approximate-budget support slide after the thank-you page, and two reference pages. A metrics slide immediately after LOOCV uses a conceptual confusion matrix to explain balanced accuracy, positive-class F1, sensitivity, specificity and precision in relation to class imbalance and triage errors; formulas and sources are preserved in the speaker notes. The rehearsal timings sum to 17 minutes and 50 seconds; the route retains its historical `18min` identifier. The budget is divided into personnel, compute, and materials subtables, with a separate total. Recorded times remain visible, while detailed assumptions are kept in the notes and [budget provenance](defensa/BUDGET.md).

The limitations slide focuses on the limited HUCA cohort without external validation and the input restricted to a rendered central V1 beat. It pairs an exterior hospital photograph with a precordial-lead illustration highlighting V1; [visual credits and adaptations](defensa/assets/referencias_visuales/README.md) document both resources. The former interpretation and open-weights blocks have been removed from that slide.

## Compile the sources

Install Tectonic or a TeX distribution with XeLaTeX and latexmk. Run the following commands from the repository root.

For the thesis:

```powershell
cd docs/tfg/memoria
tectonic main.tex
```

This produces `main.pdf`; the supplied final thesis PDF is preserved separately. Times New Roman is used when available, with TeX Gyre Termes as the alternative.

For the defense and speaker guide:

```powershell
cd docs/tfg/defensa
python build.py --compile
```

The helper accepts `--compiler PATH` for Tectonic and also detects latexmk with XeLaTeX. It writes intermediate files to `.build/` and replaces the local defense and guide PDFs. Georgia is used when available, with TeX Gyre Termes as the alternative. Microsoft fonts are not redistributed. Tectonic may download LaTeX packages on its first run.

The defense can also be compiled directly with `tectonic main.tex`. Its LaTeX content is in `content.tex`, its theme in `tema-uniovi.sty`, and its metadata and metrics in `metadata.tex` and `metrics.tex`. The theme adapts the University of Oviedo identity manual; it is not an officially approved Beamer template.

`slides.json` preserves the structured slide content and notes. `python build.py --generate` regenerates the TeX sources and overwrites direct edits to them. Ordinary compilation does not regenerate those sources. The optional `python validate.py` requires pypdf and pypdfium2.

To export the compiled defense PDF to PowerPoint:

```powershell
node export_powerpoint.mjs
```

This requires Node.js, Poppler (`pdftoppm`), and `@oai/artifact-tool`. `PDFTOPPM` can specify the executable path; `ARTIFACT_TOOL_MODULE` can specify the module's `dist/artifact_tool.mjs` file. An optional positional argument specifies the output PPTX path.

## Publication provenance

The initial deliverables and source snapshot were copied from the local `thesis/revision_nahuel/memoria/` and `thesis/revision_nahuel/defensa/` working trees. The selected thesis is the 81-page revised PDF, as explicitly requested. On 28 November 2026, the defense and guide were rebuilt and the PowerPoint re-exported to add the approximate-budget support slide, subsequently organized into subtables, and a metrics explanation after LOOCV. The thesis is preserved. [SHA256SUMS](SHA256SUMS) records the current deliverable hashes. Recompilation uses the included source snapshot and local fonts and packages, and may differ from the supplied documents.

Both main documents were compiled with Tectonic 0.17.0 before initial publication. The current thesis and defense contain 81 and 41 pages respectively. The label-distribution slide and the former slide 27 on balanced-context ICL transfer have been removed from the defense. The experimental-design diagram places VLM/ICL above CNN, with its editable Draw.io source retained; the PDF uses vector outlines for the figure text. The printed author name is Alejandro Morís Lara. Content figures have numbered captions and source credits, and the footer uses continuous slide numbers through the budget and references. Other numerical results retain their content.

The sources include chapters, appendices, front matter, generated tables, bibliography text, themes, and figure assets. Local review mappings, alternative proposals, temporary builds, and duplicate thesis PDFs are excluded.

The [defense aggregate snapshot](defensa/data/results_snapshot.json) and [numeric provenance](defensa/data/numeric_provenance.json) trace displayed metrics to `outputs/final_results_tfg/loocv/`. These copies and figures are presentation artifacts; canonical experimental results remain in `outputs/final_results_tfg/`. See [public figure and table provenance](../results.md). No clinical recordings, patient-level predictions, weights, or full experiment outputs are included.

Third-party visual credits are preserved in [the asset attribution file](defensa/assets/referencias_visuales/README.md). Repository and third-party rights remain as described in [LICENSE](../../LICENSE).
