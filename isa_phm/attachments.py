"""Discover and safely preview the Backend's ISA-JSON attachment comments."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

from .errors import DataFileError


@dataclass(frozen=True)
class Attachment:
    """A referenced dataset attachment and the ISA object it describes."""

    kind: str
    study: str
    owner_type: str
    owner: str
    reference: str
    path: Path | None
    status: str

    def preview(self, width: int = 900, height: int = 650) -> Any:
        """Return an IPython image or PDF preview for display in a notebook.

        Preview embeds the local file in notebook output; sharing a saved
        notebook may therefore share its attachment. PDF viewers vary by browser.
        """
        if self.status != "available" or self.path is None:
            raise DataFileError(f"Attachment is {self.status}: {self.reference}")
        try:
            from IPython.display import IFrame, Image
        except ImportError as exc:
            raise ImportError("Notebook previews require IPython; install isa-phm[notebooks].") from exc
        suffix = self.path.suffix.lower()
        if self.kind == "image" and suffix in {".png", ".jpg", ".jpeg", ".gif"}:
            return Image(filename=str(self.path))
        if self.kind == "datasheet" and suffix == ".pdf":
            data = base64.b64encode(self.path.read_bytes()).decode("ascii")
            return IFrame(src=f"data:application/pdf;base64,{data}", width=width, height=height)
        raise DataFileError(f"Inline preview is not supported for {self.path.suffix or 'this file type'}; use the attachment path to open it locally.")

    def display(self, width: int = 900, height: int = 650) -> None:
        """Display this attachment in IPython/Jupyter, when supported."""
        try:
            from IPython.display import display
        except ImportError as exc:
            raise ImportError("Notebook previews require IPython; install isa-phm[notebooks].") from exc
        display(self.preview(width=width, height=height))


def _attachment(kind: str, study: str, owner_type: str, owner: str, reference: str, root: Path) -> Attachment:
    folder = "Images" if kind == "image" else "Datasheets"
    raw = reference.strip()
    parts = PurePosixPath(raw.replace("\\", "/"))
    if (not raw or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw)
            or PureWindowsPath(raw).is_absolute() or parts.is_absolute()
            or not parts.parts or parts.parts[0] != folder or ".." in parts.parts):
        return Attachment(kind, study, owner_type, owner, reference, None, "unsafe")
    try:
        target = (root / Path(*parts.parts)).resolve()
        target.relative_to(root.resolve())
        target.relative_to((root / folder).resolve())
    except (OSError, RuntimeError, ValueError):
        return Attachment(kind, study, owner_type, owner, reference, None, "unsafe")
    return Attachment(kind, study, owner_type, owner, reference, target, "available" if target.is_file() else "missing")


def discover_attachments(document: dict, root: Path) -> list[Attachment]:
    """Discover only attachment locations used by the current Backend mapping."""
    result = []
    def add_comments(obj, mapping, study, owner_type, owner):
        for comment in obj.get("comments", []) or []:
            if not isinstance(comment, dict):
                continue
            kind = mapping.get(str(comment.get("name", "")).strip().casefold())
            value = comment.get("value")
            if kind and isinstance(value, str) and value.strip():
                result.append(_attachment(kind, study, owner_type, owner, value, root))

    for study in document.get("studies", []) or []:
        if not isinstance(study, dict):
            continue
        title = str(study.get("title") or study.get("identifier") or "Unnamed study")
        materials = study.get("materials") or {}
        if not isinstance(materials, dict):
            materials = {}
        categories = {
            item.get("@id"): (item.get("characteristicType") or {}).get("annotationValue")
            for item in study.get("characteristicCategories", []) or [] if isinstance(item, dict)
        }
        for material_type in ("sources", "samples"):
            for material in materials.get(material_type, []) or []:
                if not isinstance(material, dict):
                    continue
                owner = str(material.get("name") or material.get("@id") or material_type)
                if material_type == "sources":
                    add_comments(material, {"image": "image"}, title, "source", owner)
                for characteristic in material.get("characteristics", []) or []:
                    if isinstance(characteristic, dict):
                        category = characteristic.get("category") or {}
                        if not isinstance(category, dict):
                            category = {}
                        category_name = category.get("annotationValue") or categories.get(category.get("@id")) or category.get("@id") or "characteristic"
                        add_comments(characteristic, {"datasheet": "datasheet"}, title, "characteristic", f"{owner} / {category_name}")
        for assay in study.get("assays", []) or []:
            if isinstance(assay, dict):
                comments = assay.get("comments", []) or []
                alias = next((c.get("value") for name in ("sensor_alias", "sensor alias") for c in comments if isinstance(c, dict) and str(c.get("name", "")).strip().casefold() == name and c.get("value")), None)
                owner = str(alias or assay.get("filename") or assay.get("@id") or "assay")
                add_comments(assay, {"datasheet": "datasheet"}, title, "sensor", owner)
                add_comments(assay, {"physical_device_datasheet": "datasheet", "physical device datasheet": "datasheet"}, title, "physical device", owner)
    return result
