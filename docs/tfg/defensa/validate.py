#!/usr/bin/env python3
"""Validate the delivered slides, figure coverage and canonical provenance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resource_types(resources: object, seen: set) -> tuple[int, int]:
    if resources is None:
        return 0, 0
    resources = resources.get_object()
    images, forms = 0, 0
    xobjects = resources.get("/XObject")
    if xobjects is None:
        return 0, 0
    xobjects = xobjects.get_object()
    for value in xobjects.values():
        identity = (getattr(value, "idnum", None), getattr(value, "generation", None))
        if identity in seen:
            continue
        seen.add(identity)
        item = value.get_object()
        if item.get("/Subtype") == "/Image":
            images += 1
        elif item.get("/Subtype") == "/Form":
            forms += 1
            nested_images, nested_forms = resource_types(item.get("/Resources"), seen)
            images += nested_images
            forms += nested_forms
    return images, forms


def main() -> None:
    inventory = json.loads((HERE / "inventario_figuras.json").read_text(encoding="utf-8"))
    slides = json.loads((HERE / "slides.json").read_text(encoding="utf-8"))
    expected_main = (
        "p01 problema p02 a03 objetivos indice antecedentes latido p03 a15 cnn_concepto p05 p06 a04 a05 huca p04 p08 p07 a06 escenarios p09 escenario_simulador p10 a07 escenario_transferencia p12 antecedente_dominio a11 p11 a09 a10 escenario_clinico p13 p14 p16 p15 futuro cierre"
    ).split()
    main_slides = [slide for slide in slides if slide["section"] == "main"]
    assert [slide["id"] for slide in main_slides] == expected_main
    assert len(slides) == len({slide["id"] for slide in slides}) == 41
    assert sum(slide["seconds"] for slide in main_slides) == 1080
    assert len(inventory["scientific_figures"]) == 22
    assert all(not slide["figures"] for slide in slides if slide["section"] != "main")
    route = inventory["routes"]["18min"]
    assert route == [slide["id"] for slide in slides]
    for item in inventory["scientific_figures"]:
        assert sha(HERE / item["file"]) == item["sha256"], item["file"]
        for ident, page in zip(item["slides"], item["pages_18min"], strict=True):
            assert route[page - 1] == ident

    snapshot = json.loads((HERE / "data/results_snapshot.json").read_text(encoding="utf-8"))
    canonical = ROOT / "outputs/final_results_tfg/loocv"
    canonical_checked = 0
    if canonical.is_dir():
        for source in snapshot["files"].values():
            assert sha(canonical / source["path"]) == source["sha256"]
            canonical_checked += 1

    stem = "Defensa_TFG"
    path = HERE / (stem + ".pdf")
    document = pdfium.PdfDocument(str(path))
    assert len(document) == len(route)
    assert all(abs(page.get_width() / page.get_height() - 16 / 9) < 0.001 for page in document)
    texts = [page.get_textpage().get_text_range() for page in document]
    combined = "\n".join(texts)
    for number in ("0,761", "0,498"):
        assert number in combined, number
    protocol = texts[route.index("p09")]
    assert "8 validación" in protocol
    scenarios = {
        "escenario_simulador": ("Simulación", "Simulador", "Simulador"),
        "escenario_transferencia": ("Cambio de dominio", "Simulador", "HUCA"),
        "escenario_clinico": ("Evaluación clínica", "HUCA", "HUCA"),
    }
    overview = texts[route.index("escenarios")]
    headings = "Escenario Origen de ejemplos Datos de prueba"
    for ident, row in scenarios.items():
        assert row[0] in overview
        assert " ".join(texts[route.index(ident)].split()) == " ".join((headings, *row))
    morphology = " ".join(texts[route.index("latido")].split())
    assert all(label in morphology for label in ("QRS", "RBBB", "Segmento ST", "Onda T"))
    assert not any("??" in text for text in texts)
    clinical_text = texts[route.index("p13")]
    assert "0,761" in clinical_text and "0,535" in clinical_text
    assert "F1 de la clase positiva" in clinical_text
    assert "Enfermedades cardiovasculares" in texts[1]
    motivation = " ".join(texts[2].split())
    assert "Electrocardiograma en formato imagen" in motivation
    assert "interpretar el trazado con pocos ejemplos" in motivation
    assert "Propuesta y objetivos del TFG" in texts[route.index("objetivos")]
    assert "Estructura de la exposición" in texts[route.index("indice")]
    assert "Gracias por su atención" in texts[len(main_slides) - 1]
    assert all(number in clinical_text for number in ("F1", "0,761", "0,535"))
    reader = PdfReader(path)
    destinations = reader.named_destinations
    for index, ident in enumerate(route):
        assert reader.get_destination_page_number(destinations[ident]) == index
    assert reader.get_destination_page_number(destinations["conclusion"]) == len(main_slides) - 1
    forms = 0
    for slide, page in zip(slides, reader.pages, strict=True):
        _, vector_forms = resource_types(page.get("/Resources"), set())
        forms += vector_forms
        targets = set()
        for ref in page.get("/Annots", []):
            annotation = ref.get_object()
            if annotation.get("/Subtype") != "/Link":
                continue
            x0, y0, x1, y1 = map(float, annotation["/Rect"])
            assert x0 >= -1 and y0 >= -1
            assert x1 <= float(page.mediabox.width) + 1
            assert y1 <= float(page.mediabox.height) + 1
            action = annotation.get("/A", {}).get_object() if "/A" in annotation else {}
            if action.get("/S") == "/GoTo":
                target = str(action["/D"])
                assert target in destinations, target
                targets.add(target)
        expected = set(slide.get("links", {})) | set(slide.get("return_to", []))
        if slide["id"] == "indice":
            expected |= {"antecedentes", "p05", "escenarios", "escenario_simulador", "p16"}
        if slide["id"] == "escenarios":
            expected |= set(scenarios)
        assert expected <= targets, (slide["id"], expected - targets)
    assert forms > 24
    results = {
        stem: {
            "pages": len(document),
            "main_slides": len(main_slides),
            "appendix_slides": len(document) - len(main_slides),
            "vector_forms": forms,
            "sha256": sha(path),
        }
    }
    guide = pdfium.PdfDocument(str(HERE / "Guion_defensa.pdf"))
    assert len(guide) >= 8
    guide_text = "\n".join(page.get_textpage().get_text_range() for page in guide)
    assert all(number in guide_text for number in ("0,741", "0,781", "ocho pacientes"))
    results["Guion_defensa"] = {"pages": len(guide), "sha256": sha(HERE / "Guion_defensa.pdf")}
    for source in ("main", "guion"):
        log = HERE / ".build" / source / (source + ".log")
        if not log.exists():
            continue
        content = log.read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"Overfull|undefined references|LaTeX Error", content), log

    protected_path = HERE / ".build/protected_hashes.json"
    protected_count = 0
    if protected_path.is_file():
        protected = json.loads(protected_path.read_text(encoding="utf-8"))
        for relative, expected in protected.items():
            assert sha(ROOT / relative) == expected, relative
            protected_count += 1
    result = {
        "status": "passed",
        "scientific_figures": len(inventory["scientific_figures"]),
        "narrative_order_verified": True,
        "concepts_before_implementations": True,
        "canonical_aggregate_files_verified": canonical_checked,
        "protected_files_unchanged": protected_count,
        "presentations_and_guide": results,
        "notes": [
            "Durations are rehearsal targets; the current tribunal limit is not confirmed.",
            "All inventoried scientific figures appear in the main route.",
            "The original ICL scheme and published visual references may contain raster elements.",
            "No per-patient predictions or identifiers are packaged.",
        ],
    }
    (HERE / "validacion.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
