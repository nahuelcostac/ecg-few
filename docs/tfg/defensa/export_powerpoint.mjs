import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const directory = fileURLToPath(new URL(".", import.meta.url));
const output = resolve(process.argv[2] ?? join(directory, "Defensa_TFG.pptx"));
const artifactToolPath = process.env.ARTIFACT_TOOL_MODULE;
const { Presentation, PresentationFile } = await import(
  artifactToolPath ? pathToFileURL(resolve(artifactToolPath)).href : "@oai/artifact-tool"
);
const slides = JSON.parse(await readFile(join(directory, "slides.json"), "utf8"));
const metrics = await readFile(join(directory, "metrics.tex"), "utf8");
const macros = new Map(
  [...metrics.matchAll(/\\newcommand\{\\(\w+)\}\{([^{}]+)\}/g)].map(
    ([, name, value]) => [name, value],
  ),
);

const slideSize = { width: 1280, height: 720 };
const presentation = Presentation.create({ slideSize });
const renderDirectory = await mkdtemp(join(tmpdir(), "defensa-tfg-"));

try {
  const prefix = join(renderDirectory, "slide");
  execFileSync(process.env.PDFTOPPM ?? "pdftoppm", [
    "-png", "-scale-to-x", "1920", "-scale-to-y", "1080",
    join(directory, "Defensa_TFG.pdf"), prefix,
  ], { stdio: "inherit" });

  for (const [index, source] of slides.entries()) {
    const number = String(index + 1).padStart(String(slides.length).length, "0");
    const slide = presentation.slides.add();
    slide.images.add({
      blob: await readFile(`${prefix}-${number}.png`),
      contentType: "image/png",
      alt: source.title,
      fit: "contain",
      position: { left: 0, top: 0, ...slideSize },
    });
    slide.speakerNotes.text = source.notes
      .replace(/\\(\w+)\{\}/g, (original, name) => macros.get(name) ?? original)
      .replace(/\\href\{([^{}]+)\}\{([^{}]+)\}/g, "$2 ($1)");
  }

  const deck = await PresentationFile.exportPptx(presentation);
  await deck.save(output);
} finally {
  await rm(renderDirectory, { recursive: true, force: true });
}
