"""Keep figure legends outside the data area in thesis presentation assets."""

from __future__ import annotations

from matplotlib.figure import Figure


def external_legend(fig: Figure) -> None:
    """Replace axes legends with one deduplicated legend below the figure."""
    entries = {}
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        entries.update(zip(labels, handles, strict=True))
        legend.remove()
    fig.tight_layout(pad=0.55 if entries else 0.35)
    if entries:
        fig.legend(
            entries.values(),
            entries.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=min(2, len(entries)),
            frameon=False,
            fontsize=7,
            handlelength=1.8,
            columnspacing=1.2,
        )
