"""Conservative metadata capability resolution for ISA-PHM analyses.

The functions here report what can safely be attempted from the metadata that
is present.  They never invent values or make missing optional metadata a
dataset-validation error.
"""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import (
    AnalysisCapabilities,
    AssayAnalysisCapabilities,
    AssayModel,
    Capability,
    MetadataResolution,
    ParameterValue,
    RunRecord,
)


_FREQUENCY_UNITS = {"hz": 1.0, "khz": 1_000.0, "mhz": 1_000_000.0}


def _normalise_unit(unit: str | None) -> str:
    return (unit or "").strip().lower().replace(" ", "")


def _number(value: object) -> float | None:
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def resolve_sampling_frequency(params: Iterable[ParameterValue]) -> MetadataResolution:
    """Resolve an explicit sampling-frequency parameter, if one is present."""
    matches: list[tuple[float, ParameterValue]] = []
    invalid_names: list[str] = []
    statuses: set[str] = set()
    for parameter in params:
        if " ".join(parameter.parameter_name.casefold().split()) != "sampling rate":
            continue
        if isinstance(parameter.value, str) and parameter.value.strip().lower() in {"unknown", "not applicable"}:
            statuses.add(parameter.value.strip().lower())
            continue
        multiplier = _FREQUENCY_UNITS.get(_normalise_unit(parameter.unit))
        value = _number(parameter.value)
        if multiplier is None or value is None:
            invalid_names.append(parameter.parameter_name)
            continue
        matches.append((value * multiplier, parameter))

    if statuses:
        if matches or invalid_names or len(statuses) > 1:
            return MetadataResolution(available=False, reason="Conflicting sampling-frequency values or statuses were provided.")
        return MetadataResolution(available=False, reason=f"Sampling frequency is {next(iter(statuses))}.")
    if not matches:
        reason = (
            "Sampling frequency is not provided as a recognised measurement parameter."
            if not invalid_names
            else "Sampling frequency has an invalid value or non-frequency unit."
        )
        return MetadataResolution(available=False, reason=reason)

    values = {value for value, _ in matches}
    if len(values) != 1:
        return MetadataResolution(
            available=False,
            reason="Conflicting sampling-frequency values were provided for this run.",
        )
    value, parameter = matches[0]
    return MetadataResolution(
        available=True,
        value=value,
        unit="Hz",
        parameter_name=parameter.parameter_name,
        source="measurement_parameter",
    )


def resolve_measured_unit(params: Iterable[ParameterValue]) -> MetadataResolution:
    """Resolve a declared measured unit without converting or standardising it."""
    candidates: list[tuple[str, ParameterValue]] = []
    statuses: set[str] = set()
    for parameter in params:
        name = " ".join(parameter.parameter_name.casefold().split())
        if name != "measured unit":
            continue
        if isinstance(parameter.value, str) and parameter.value.strip().lower() in {"unknown", "not applicable"}:
            statuses.add(parameter.value.strip().lower())
            continue
        if isinstance(parameter.value, str) and parameter.value.strip():
            candidates.append((parameter.value.strip(), parameter))

    if statuses:
        if candidates or len(statuses) > 1:
            return MetadataResolution(available=False, reason="Conflicting measured-unit values or statuses were provided.")
        return MetadataResolution(available=False, reason=f"Measured unit is {next(iter(statuses))}.")
    if not candidates:
        return MetadataResolution(
            available=False,
            reason="Measured unit is not provided in the measurement metadata.",
        )
    values = {value for value, _ in candidates}
    if len(values) != 1:
        return MetadataResolution(
            available=False,
            reason="Conflicting measured-unit values were provided for this run.",
        )
    value, parameter = candidates[0]
    return MetadataResolution(
        available=True,
        value=value,
        unit=value,
        parameter_name=parameter.parameter_name,
        source="measurement_parameter",
    )


def run_analysis_capabilities(run: RunRecord) -> AnalysisCapabilities:
    """Return the analyses supported by one run's files and metadata."""
    has_data = any(data_file and data_file.exists for data_file in (run.raw_file, run.processed_file))
    data_loading = Capability(
        available=has_data,
        reason=None if has_data else "No accessible raw or processed data file is linked to this run.",
    )
    sampling_frequency = resolve_sampling_frequency(run.measurement_params)
    measured_unit = resolve_measured_unit(run.measurement_params)
    return AnalysisCapabilities(
        run_id=run.run_id,
        data_loading=data_loading,
        time_domain_features=Capability(
            available=has_data,
            reason=data_loading.reason,
        ),
        frequency_domain=Capability(
            available=has_data and sampling_frequency.available,
            reason=(None if has_data and sampling_frequency.available else sampling_frequency.reason or data_loading.reason),
        ),
        physical_unit_interpretation=Capability(
            available=measured_unit.available,
            reason=measured_unit.reason,
        ),
        sampling_frequency=sampling_frequency,
        measured_unit=measured_unit,
    )


def assay_analysis_capabilities(assay: AssayModel) -> AssayAnalysisCapabilities:
    """Summarise the capabilities shared by every run in an assay."""
    runs = [run_analysis_capabilities(run) for run in assay.runs]

    def shared_capability(attribute: str) -> Capability:
        if not runs:
            return Capability(available=False, reason="This assay has no runs.")
        if all(getattr(run, attribute).available for run in runs):
            return Capability(available=True)
        return Capability(
            available=False,
            reason=f"One or more runs cannot support {attribute.replace('_', ' ')}.",
        )

    def shared_metadata(attribute: str) -> MetadataResolution:
        if not runs:
            return MetadataResolution(available=False, reason="This assay has no runs.")
        resolutions = [getattr(run, attribute) for run in runs]
        if not all(item.available for item in resolutions):
            return MetadataResolution(
                available=False,
                reason=f"One or more runs do not provide {attribute.replace('_', ' ')}.",
            )
        values = {item.value for item in resolutions}
        if len(values) != 1:
            return MetadataResolution(
                available=False,
                reason=f"Runs provide conflicting {attribute.replace('_', ' ')} values.",
            )
        return resolutions[0]

    sampling_frequency = shared_metadata("sampling_frequency")
    measured_unit = shared_metadata("measured_unit")
    frequency_domain = shared_capability("frequency_domain")
    if not sampling_frequency.available:
        frequency_domain = Capability(available=False, reason=sampling_frequency.reason)
    physical_unit = shared_capability("physical_unit_interpretation")
    if not measured_unit.available:
        physical_unit = Capability(available=False, reason=measured_unit.reason)
    return AssayAnalysisCapabilities(
        assay_id=assay.assay_id,
        data_loading=shared_capability("data_loading"),
        time_domain_features=shared_capability("time_domain_features"),
        frequency_domain=frequency_domain,
        physical_unit_interpretation=physical_unit,
        sampling_frequency=sampling_frequency,
        measured_unit=measured_unit,
        runs=runs,
    )
