#!/usr/bin/env python3
"""Compile the portable Beamer deck and speaker guide with Tectonic or latexmk.

Use --refresh only inside the original repository to copy canonical aggregates
and refresh metadata and figures. Ordinary builds use the included snapshot.
"""

# ruff: noqa: E501 -- Speaker-guide paragraphs are preserved as readable LaTeX prose.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = "google/gemma-4-E4B-it"
SOURCES = {
    "cnn_sim": "cnn_simulator_qrs/cnn_summary_by_k.csv",
    "cnn_transfer": "cnn/cnn_summary_by_k.csv",
    "vlm_sim": "vlm_simulator_qrs/vlm_summary_by_model_condition_k.csv",
    "vlm_transfer": "vlm/vlm_summary_by_model_condition_k.csv",
    "vlm_real": "vlm_real_context/vlm_summary_by_model_condition_k.csv",
    "cnn_real": "cnn_clinical/balanced/cnn_clinical_summary_by_k.csv",
    "cnn_standard": "cnn_clinical/estandar/cnn_clinical_summary_by_k.csv",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def refresh() -> None:
    canonical = ROOT / "outputs/final_results_tfg/loocv"
    snapshot = {"source": "outputs/final_results_tfg/loocv", "files": {}}
    for key, relative in SOURCES.items():
        path = canonical / relative
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        # Only aggregate fields enter the portable package.
        keep = (
            "k",
            "condition",
            "model",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "sensitivity_mean",
            "specificity_mean",
            "f1_mean",
            "f1_std",
            "tp",
            "tn",
            "fp",
            "fn",
            "n_runs",
            "n_patients",
        )
        rows = [{k: v for k, v in row.items() if k in keep} for row in rows]
        snapshot["files"][key] = {"path": relative, "sha256": sha(path), "rows": rows}
    write(HERE / "data/results_snapshot.json", json.dumps(snapshot, ensure_ascii=False, indent=2))
    shutil.copy2(ROOT / "thesis/thesis/config/metadata.tex", HERE / "metadata.tex")
    for directory in ("results", "sources"):
        for path in (ROOT / "thesis/thesis/assets" / directory).glob("*.pdf"):
            shutil.copy2(path, HERE / "assets" / path.name)
    metrics()
    crop_panels()


def metrics() -> None:
    data = json.loads((HERE / "data/results_snapshot.json").read_text(encoding="utf-8"))

    def row(key: str, k: int, condition: str | None = None) -> dict:
        return next(
            r
            for r in data["files"][key]["rows"]
            if int(r["k"]) == k
            and (condition is None or r.get("condition") == condition)
            and (not key.startswith("vlm") or r.get("model", MODEL) == MODEL)
        )

    requests = {
        "SimCnnBA": ("cnn_sim", 32, None, "balanced_accuracy_mean"),
        "SimVlmBA": ("vlm_sim", 32, "balanced", "balanced_accuracy_mean"),
        "TransferCnnBestBA": ("cnn_transfer", 4, None, "balanced_accuracy_mean"),
        "TransferCnnLastBA": ("cnn_transfer", 32, None, "balanced_accuracy_mean"),
        "TransferVlmBestBA": ("vlm_transfer", 16, "balanced", "balanced_accuracy_mean"),
        "RealVlmBA": ("vlm_real", 32, "balanced", "balanced_accuracy_mean"),
        "RealVlmSens": ("vlm_real", 32, "balanced", "sensitivity_mean"),
        "RealVlmSpec": ("vlm_real", 32, "balanced", "specificity_mean"),
        "RealVlmFone": ("vlm_real", 32, "balanced", "f1_mean"),
        "RealCnnBestBA": ("cnn_real", 16, "balanced", "balanced_accuracy_mean"),
        "RealCnnLastBA": ("cnn_real", 32, "balanced", "balanced_accuracy_mean"),
    }
    lines = ["% Generated from the included canonical aggregate snapshot."]
    provenance = {}
    for name, (key, k, condition, column) in requests.items():
        selected = row(key, k, condition)
        value = float(selected[column])
        formatted = f"{value:.3f}".replace(".", ",")
        lines.append("\\newcommand{\\" + name + "}{" + formatted + "}")
        provenance[name] = {
            "source": "outputs/final_results_tfg/loocv/" + data["files"][key]["path"],
            "k": k,
            "condition": condition,
            "column": column,
            "value": value,
            "source_sha256": data["files"][key]["sha256"],
        }
    for prefix, key, condition in (
        ("Sim", "vlm_sim", "balanced"),
        ("Huca", "vlm_real", "balanced"),
    ):
        selected = row(key, 32, condition)
        runs = int(selected["n_runs"])
        positive = (int(selected["tp"]) + int(selected["fn"])) // runs
        negative = (int(selected["tn"]) + int(selected["fp"])) // runs
        for suffix, value in (
            ("Positive", positive),
            ("Negative", negative),
            ("Patients", positive + negative),
        ):
            name = prefix + suffix
            lines.append("\\newcommand{\\" + name + "}{" + str(value) + "}")
            provenance[name] = {
                "source": data["files"][key]["path"],
                "value": value,
                "calculation": "Class totals in aggregate confusion counts divided by n_runs",
            }
    assert provenance["HucaPatients"]["value"] == 317
    assert provenance["SimPatients"]["value"] == 100
    write(HERE / "metrics.tex", "\n".join(lines) + "\n")
    write(
        HERE / "data/numeric_provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2)
    )


def crop_panels() -> None:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject

    specifications = {
        "clinical_ba.pdf": ("clinical_cnn_vs_gemma_performance.pdf", (0, 184, 436.509375, 335)),
        "clinical_f1.pdf": ("clinical_cnn_vs_gemma_performance.pdf", (0, 0, 436.509375, 181)),
        "clinical_legend.pdf": ("clinical_cnn_vs_gemma_performance.pdf", (0, 356, 436.509375, 384)),
        "clinical_xaxis.pdf": ("clinical_cnn_vs_gemma_performance.pdf", (0, 4, 436.509375, 33)),
        "context_balanced.pdf": ("domain_gap_icl_context_ba_by_k.pdf", (0, 0, 409.9525, 230.5)),
        "context_standard.pdf": (
            "domain_gap_icl_context_ba_by_k.pdf",
            (0, 235, 409.9525, 400.895125),
        ),
        "context_xaxis.pdf": ("domain_gap_icl_context_ba_by_k.pdf", (0, 37, 409.9525, 64)),
        "transfer_ba.pdf": (
            "transfer_controls_synthetic_to_real_summary.pdf",
            (0, 217, 496.40625, 371.014515625),
        ),
        "transfer_f1.pdf": (
            "transfer_controls_synthetic_to_real_summary.pdf",
            (0, 0, 496.40625, 215),
        ),
        "transfer_legend.pdf": (
            "transfer_controls_synthetic_to_real_summary.pdf",
            (0, 0, 496.40625, 55),
        ),
    }
    for target, (source, box) in specifications.items():
        reader = PdfReader(HERE / "assets" / source)
        page = reader.pages[0]
        page.mediabox = RectangleObject(box)
        page.cropbox = RectangleObject(box)
        writer = PdfWriter()
        writer.add_page(page)
        with (HERE / "assets" / target).open("wb") as handle:
            writer.write(handle)
    write(HERE / "data/vector_crops.json", json.dumps(specifications, ensure_ascii=False, indent=2))


def generate_sources() -> None:
    data = json.loads((HERE / "slides.json").read_text(encoding="utf-8"))
    full = [slide["id"] for slide in data if slide["section"] == "main"]
    support = [slide["id"] for slide in data if slide["section"] == "appendix"]
    sections = []
    for slide in data:
        label = slide["id"]
        body = slide["latex"]
        notes = slide["notes"]
        navigation = [
            rf"\hyperlink{{{target}}}{{{caption}}}"
            for target, caption in slide.get("links", {}).items()
        ]
        origins = slide.get("return_to", [])
        for origin in origins:
            caption = (
                "Volver a la diapositiva"
                if len(origins) == 1
                else f"Volver a la diapositiva {full.index(origin) + 1}"
            )
            navigation.append(rf"\hyperlink{{{origin}}}{{{caption}}}")
        if slide["section"] == "appendix" and not origins:
            navigation.append(r"\hyperlink{conclusion}{Volver al cierre}")
        footer = r"\quad ".join(navigation) or "Defensa del TFG"
        plain = label in {"p01", "p02", "cierre"} or slide.get("plain", False)
        prefix = "[plain]" if plain else "[t]"
        sections.append(
            r"\expandafter\newcommand\csname Slide"
            + slide["id"]
            + r"\endcsname{%"
            + "\n"
            + r"\renewcommand{\defenseblock}{"
            + slide.get("block", "Anexos")
            + "}\n"
            + rf"\renewcommand{{\defensenav}}{{{footer}}}"
            + "\n"
            + r"\begin{frame}"
            + prefix
            + r"\hypertarget{"
            + label
            + r"}{}"
            + "\n"
            + (r"\hypertarget{conclusion}{}" + "\n" if label == full[-1] else "")
            + ("" if plain else r"\frametitle{" + slide["title"] + "}\n")
            + body
            + "\n"
            + r"\note{"
            + notes
            + "}\n"
            + r"\end{frame}}"
            + "\n"
        )
    write(HERE / "content.tex", "\n".join(sections))
    text = (
        "% !TEX program = xelatex\n"
        + r"\documentclass[aspectratio=169,professionalfonts]{beamer}"
        + "\n"
        + r"\usepackage{tema-uniovi}"
        + "\n"
        + r"\input{metadata}\input{metrics}\input{content}"
        + "\n"
        + r"\hypersetup{pdftitle={\ThesisTitle},pdfauthor={\AuthorName},pdfsubject={Defensa del TFG}}\title{\ThesisTitle}"
        + "\n"
        + r"\begin{document}"
        + "\n"
        + "\n".join(r"\csname Slide" + ident + r"\endcsname" for ident in full)
        + "\n"
        + r"\appendix\defenseappendixtrue"
        + "\n"
        + "\n".join(r"\csname Slide" + ident + r"\endcsname" for ident in support)
        + "\n"
        + r"\end{document}"
        + "\n"
    )
    write(HERE / "main.tex", text)
    guide(data, full, support)
    inventory(data, full + support)


def guide(data: list[dict], full: list[str], annex: list[str]) -> None:
    header = r"""% !TEX program = xelatex
\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\IfFontExistsTF{Georgia}{\setmainfont{Georgia}}{\setmainfont{TeX Gyre Termes}}
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage[margin=23mm]{geometry}
\usepackage{parskip,enumitem,xcolor,xurl,hyperref,fancyhdr,longtable,booktabs,needspace}
\input{metadata}\input{metrics}
\hypersetup{hidelinks,pdftitle={Guion de defensa del TFG},pdfauthor={\AuthorName},pdfsubject={\ThesisTitle}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Guion de defensa del TFG}\fancyhead[R]{\small Alejandro Moris Lara}\fancyfoot[C]{\thepage}
\setlength{\headheight}{15pt}
\begin{document}
\begin{center}{\LARGE Guion para la defensa del TFG}\par\medskip
{\large\ThesisTitle}\par\medskip\AuthorName\par Noviembre de 2026\end{center}
\section*{Preparación y duración}
La defensa contiene @FULLCOUNT@ diapositivas principales para un ensayo de 18 minutos y dos páginas de referencias.

La apertura relaciona la escasez de positivos con la representación visual y presenta el aprendizaje en contexto antes de los objetivos. Después del estado del arte se explica el latido y se sitúan los aspectos éticos. Tras las implementaciones se presentan los tres escenarios y el protocolo por paciente. Una diapositiva de transición abre cada bloque de resultados.

La referencia pública localizada para la EPI indica entre 15 y 20 minutos de forma orientativa y deja la duración al tribunal. El procedimiento del curso 2026--2027 está enlazado en SharePoint y requiere acceso institucional. Por tanto, estos tiempos son objetivos de ensayo, no una confirmación del límite de tu tribunal.

Ensaya en voz alta, siguiendo el gráfico que ve el tribunal. No leas las diapositivas ni recites los decimales. Al explicar una curva, identifica primero el eje horizontal, la métrica y las series, y después señala el cambio relevante. Las barras muestran variación entre semillas, no intervalos de confianza.

La presentación es una adaptación a 16:9 de los modelos del apartado 6.1 del Manual de Identidad Corporativa de UniOvi. No constituye una plantilla Beamer publicada o aprobada por la institución. Se conservan los emblemas originales, con Georgia para los títulos y el texto. Si Georgia no está instalada, el proyecto utiliza TeX Gyre Termes como alternativa.

\clearpage\section*{Ruta principal: 18 minutos}
\begin{longtable}{p{12mm}p{112mm}r}\toprule Diap. & Contenido & Acumulado\\\midrule\endhead
"""
    lookup = {s["id"]: s for s in data}
    lines = [header.replace("@FULLCOUNT@", str(len(full)))]
    elapsed = 0
    for i, ident in enumerate(full, 1):
        slide = lookup[ident]
        elapsed += slide["seconds"]
        lines.append(f"{i} & {slide['title']} & {elapsed // 60}:{elapsed % 60:02d}\\\\\n")
    lines.append(r"\bottomrule\end{longtable}" + "\n")
    for i, ident in enumerate(full, 1):
        slide = lookup[ident]
        lines.append(
            r"\Needspace{8\baselineskip}\section*{" + f"{i:02d}. " + slide["title"] + "}\n"
        )
        lines.append(r"\textbf{Tiempo orientativo: }" + str(slide["seconds"]) + " segundos.\n\n")
        lines.append(slide["notes"] + "\n\n")
        if slide["transition"]:
            lines.append(r"\textbf{Transición.} " + slide["transition"] + "\n\n")
    lines.append(r"\clearpage\section*{Referencias y preguntas previsibles}" + "\n")
    for ident in annex:
        slide = lookup[ident]
        lines.append(
            r"\Needspace{7\baselineskip}\subsection*{"
            + slide["title"]
            + "}\n"
            + slide["notes"]
            + "\n\n"
        )
    lines.append(r"\section*{Preguntas que conviene preparar}" + "\n")
    lines.append(r"""\begin{description}[style=nextline]
\item[¿La comparación utiliza el mismo presupuesto de etiquetas?] El valor de K es común, pero la CNN con HUCA dispone además de ocho pacientes de validación. Se debe declarar esta diferencia al explicar el protocolo.
\item[¿Por qué el simulador no transfiere bien?] Cambian tanto la apariencia de las imágenes como el significado de la etiqueta. El experimento no separa ambos efectos.
\item[¿La ventaja de Gemma demuestra significación estadística?] No. Las barras representan la desviación entre tres semillas y no sustituyen un contraste estadístico ni una validación externa.
\item[¿Podría usarse ya para descartar pacientes?] La mejor configuración todavía omite positivos. La utilidad como apoyo a la priorización necesita evaluación clínica, externa y prospectiva.
\item[¿Qué demuestra Grad-CAM?] Muestra regiones asociadas a una salida. No demuestra por sí solo un mecanismo causal ni que la CNN reconozca un criterio clínico.
\item[¿Qué ocurre sin demostraciones?] Con estas instrucciones, Gemma no identifica positivos. Es un resultado de la configuración evaluada y no una afirmación universal sobre todos los modelos o prompts.
\end{description}
\clearpage\section*{Fuentes para la preparación}
\begin{itemize}
\item Manual de Identidad Corporativa de UniOvi, apartado 6.1, páginas 57--58. \url{https://www.uniovi.es/conocenos/uniovi/identidad}
\item Página de TFG de la EPI. \url{https://epigijon.uniovi.es/estudiantes/tfg}
\item Copia pública de las directrices de entrega y defensa de la EPI, utilizada únicamente como orientación temporal. \url{https://expydoc.com/doc/6397600/directrices-para-los-estudiantes}
\item MIT Communication Lab, Slide Design. \url{https://mitcommlab.mit.edu/aeroastro/commkit/slide-design/}
\item Beamer User Guide, capítulo 5. \url{https://ctan.org/pkg/beamer}
\end{itemize}
\end{document}
""")
    write(HERE / "guion.tex", "".join(lines))


def inventory(data: list[dict], full: list[str]) -> None:
    original = sorted(
        p
        for p in (HERE / "assets").glob("*.pdf")
        if not p.name.startswith("logo_")
        and p.name
        not in set(json.loads((HERE / "data/vector_crops.json").read_text(encoding="utf-8")))
    )
    figures = []
    for p in original:
        occurrences = [s["id"] for s in data if p.name in s.get("figures", [])]
        if not occurrences:
            continue
        figures.append(
            {
                "file": "assets/" + p.name,
                "sha256": sha(p),
                "slides": occurrences,
                "pages_18min": [full.index(s) + 1 for s in occurrences],
                "format": "PDF with raster elements"
                if p.name == "icl_general_dong2024.pdf"
                else "vector PDF",
            }
        )
    write(
        HERE / "inventario_figuras.json",
        json.dumps(
            {
                "scientific_figures": figures,
                "unique_count": len(figures),
                "main_slide_count": {"18min": sum(s["section"] == "main" for s in data)},
                "routes": {"18min": full},
                "narrative_order": [
                    {
                        "slide": s["id"],
                        "title": s["title"],
                        "block": s.get("block"),
                        "seconds": s["seconds"],
                    }
                    for s in data
                    if s["section"] == "main"
                ],
                "numeric_provenance": "data/numeric_provenance.json",
                "institutional_design": "Adaptation of UniOvi corporate identity manual, section 6.1, to 16:9; not an officially published Beamer theme",
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def find_compiler(explicit: str | None) -> tuple[str, str]:
    candidates = [
        explicit,
        shutil.which("tectonic"),
        str(ROOT / ".cache/tutor-tectonic/tectonic.exe"),
        os.environ.get("CODEX_TECTONIC_PATH"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                subprocess.run(
                    [candidate, "--version"], check=True, capture_output=True, timeout=10
                )
                return "tectonic", candidate
            except (OSError, subprocess.SubprocessError):
                pass
    latexmk = shutil.which("latexmk")
    if latexmk:
        return "latexmk", latexmk
    raise RuntimeError(
        "Install Tectonic or a TeX Live distribution with latexmk and XeLaTeX, or pass --compiler PATH."
    )


def compile_all(explicit: str | None) -> None:
    kind, executable = find_compiler(explicit)
    build = HERE / ".build"
    build.mkdir(exist_ok=True)
    documents = (
        ("main", "Defensa_TFG"),
        ("guion", "Guion_defensa"),
    )
    for source, output in documents:
        directory = build / source
        directory.mkdir(exist_ok=True)
        if kind == "tectonic":
            command = [
                executable,
                source + ".tex",
                "--outdir",
                str(directory),
                "--keep-logs",
                "--keep-intermediates",
                "--reruns",
                "2",
            ]
        else:
            command = [
                executable,
                "-xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-outdir=" + str(directory),
                source + ".tex",
            ]
        log_path = directory / "build_console.txt"
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=HERE, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log_path.read_text(encoding="utf-8", errors="replace")[-9000:])
            raise RuntimeError(f"Compilation failed: {source}; see {log_path}")
        shutil.copy2(directory / (source + ".pdf"), HERE / (output + ".pdf"))
        print(f"Compiled {output}.pdf", flush=True)


def package() -> None:
    target = HERE / "Fuentes_defensa_TFG.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(HERE.rglob("*")):
            if (
                not p.is_file()
                or p == target
                or any(x in {".build", "__pycache__"} for x in p.parts)
            ):
                continue
            archive.write(p, "defensa_tfg/" + p.relative_to(HERE).as_posix())
    print(f"Packaged {target.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--compiler")
    args = parser.parse_args()
    if args.refresh:
        parser.error("Esta edición conserva los recursos de la revisión. Use --generate --compile.")
    if args.generate:
        generate_sources()
    if args.compile:
        compile_all(args.compiler)
    if args.package:
        package()
    if not any((args.refresh, args.generate, args.compile, args.package)):
        parser.print_help()


if __name__ == "__main__":
    main()
