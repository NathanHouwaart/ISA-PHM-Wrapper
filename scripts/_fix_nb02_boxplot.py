"""Fix the section 6 cross-sensor boxplot cells in notebook 02."""
import json
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "02_diagnostic_workflow.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

for c in nb["cells"]:
    cid = c.get("id", "")

    # Markdown cell — section 6 description
    if cid == "3d6b9860":
        c["source"] = [
            "## 6. Cross-sensor comparison \u2014 interactive boxplot\n",
            "\n",
            "A **Bokeh** box plot with one box per sensor channel \u2014 all channels on a single figure.\n",
            "\n",
            "**What you can do:**\n",
            "- **Hover** over a box \u2192 tooltip shows median, mean, Q1, Q3, and whisker bounds\n",
            "- **Scroll** to zoom in/out on the y-axis\n",
            "- **Drag** to pan; use *Box Zoom* in the toolbar for a precise region\n",
            "- **Click** the disc icon in the toolbar to save as PNG\n",
            "\n",
            "**How to read the boxes:**\n",
            "- Dark blue = Q1 \u2192 median (lower half of the IQR)\n",
            "- Light blue = median \u2192 Q3 (upper half of the IQR)\n",
            "- Whiskers extend to 1.5 \u00d7 IQR (or data min/max, whichever comes first)\n",
            "- **Orange dot** = mean\n",
            "\n",
            "Only summary statistics are held in memory after loading \u2014 not the full arrays.\n",
        ]
        print("Fixed markdown", cid)

    # Code cell — section 6 code
    if cid == "552a8815":
        c["source"] = [
            "# Cross-sensor boxplot \u2014 all sensors on a single interactive figure.\n",
            "fig = study.plot_sensor_boxplot(\n",
            "    file_type='raw',\n",
            "    outlier_method='fixed',\n",
            "    outlier_upper=1e7,\n",
            "    outlier_strategy='drop',\n",
            ")\n",
            "bokeh_show(fig)",
        ]
        c["outputs"] = []
        c["execution_count"] = None
        print("Fixed code", cid)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved.")
