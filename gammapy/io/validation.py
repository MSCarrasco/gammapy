# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Format validation of serialized data products.

Checks a table, a header or a FITS HDU against the definitions registered for a
data format in ``gammapy.io.registries`` (``DATA_FORMATS_MODELS``):

* ``DefinitionValidator``: one header (keywords) or one table (columns) against
  one definition;
* ``FormatValidator``: resolves the HDU class and the definitions for
  (format, version, class key), runs the header and column checks and returns a
  ``ValidationReport``;
* ``validate_hdus``: batch validation of HDUs in files, each file opened once.

Validation works on the serialized representation and builds no Gammapy product.
This module depends on ``gammapy.io.registries`` only.
"""

import logging
from dataclasses import dataclass, field
import re
from typing import NamedTuple, Optional

import numpy as np
from astropy.io import fits
import astropy.units as u

from gammapy.io.registries import (
    DATA_FORMATS_MODELS,
    DEFAULT_DATA_FORMAT,
    DEFAULT_DATA_FORMAT_VERSION,
)
from gammapy.utils.scripts import make_path

__all__ = [
    "FormatComplianceError",
    "ColumnDescription",
    "DefinitionValidator",
    "FormatValidator",
    "ValidationReport",
    "columns_from_header",
    "columns_from_table",
    "format_class_keys",
    "header_definition",
    "validate_hdus",
]

log = logging.getLogger(__name__)

# TFORMn: repeat count and type code, e.g. "6E", "1PE(100)"
_TFORM = re.compile(r"^\s*(\d*)\s*([A-Za-z])")


# --------------- HELPERS ---------------
def _value(meta, key):
    """Keyword value, with empty values ("" / [] / None) treated as absent."""
    val = meta.get(key)
    if val is None or (isinstance(val, (str, list)) and len(val) == 0):
        return None
    return val


def _hdu_class_key(meta):
    """Generic HDU class key. Format modules may register an override."""
    key = meta.get("HDUCLAS4") or meta.get("HDUCLAS1") or meta.get("EXTNAME")
    if key and "BANDS" in key:
        return "BANDS"
    return key.upper() if key else "UNKNOWN"


def header_definition(format, version, class_key):
    """Keyword definition for (format, version, HDU class key); {} if none."""
    headers = DATA_FORMATS_MODELS.get(format, {}).get("HEADER", {}).get(version, {})
    return headers.get(class_key) or headers.get("BASE") or {}


# --------------- DEFINITION VALIDATOR ---------------
_FITS_DTYPES = {
    "str": (str,),
    "int": (int, np.integer),
    "float": (float, int, np.floating, np.integer),
    "bool": (bool, np.bool_),
}


def _has_dtype(value, dtype):
    dtypes = (dtype,) if isinstance(dtype, str) else tuple(dtype)
    for name in dtypes:
        if name in ("int", "float") and isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, _FITS_DTYPES[name]):
            return True
    return False


def _check_keyword(name, value, spec):
    """Errors of one header keyword value against its spec (dtype, allowed)."""
    errors = []
    dtype = spec.get("dtype")
    if dtype and not _has_dtype(value, dtype):
        errors.append(f"Invalid {name}={value!r}: expected {dtype}")
    allowed = spec.get("allowed")
    if allowed is not None and value not in allowed:
        errors.append(f"Invalid {name}={value!r}: allowed values {list(allowed)}")
    return errors


class ColumnDescription(NamedTuple):
    """What the column checks need to know about a column: per-row ndim and unit."""

    ndim: int
    unit: Optional[u.UnitBase]


def columns_from_table(table):
    """Column descriptions of an in-memory `~astropy.table.Table`."""
    return {
        name: ColumnDescription(ndim=table[name].ndim - 1, unit=table[name].unit)
        for name in table.colnames
    }


def _tdim(value):
    """Parse a TDIMn value such as ``"(3,2)"`` into a tuple of ints."""
    return tuple(int(x) for x in str(value).strip().strip("()").split(",") if x.strip())


def columns_from_header(header):
    """Column descriptions of a binary table, read from its FITS header only.

    Uses the ``TFIELDS``, ``TTYPEn``, ``TFORMn``, ``TDIMn`` and ``TUNITn``
    keywords, so no table data is read. The per-row ``ndim`` and the unit are
    those `~astropy.table.Table.read` gives the column.
    """
    columns = {}
    for idx in range(1, int(header.get("TFIELDS", 0)) + 1):
        name = header.get(f"TTYPE{idx}")
        if name is None:
            continue
        match = _TFORM.match(str(header.get(f"TFORM{idx}", "")))
        repeat = int(match.group(1) or 1) if match else 1
        code = match.group(2).upper() if match else ""
        tdim = header.get(f"TDIM{idx}")
        if code in ("P", "Q"):  # variable-length array: one object per row
            ndim = 0
        elif code == "A":  # strings: TDIM includes the string length
            ndim = len(_tdim(tdim)) - 1 if tdim else 0
        elif code == "X":  # bits are read as a boolean array
            ndim = 1
        elif tdim:
            ndim = len(_tdim(tdim))
        else:
            ndim = 1 if repeat > 1 else 0
        tunit = header.get(f"TUNIT{idx}")
        unit = (
            u.Unit(tunit, format="fits", parse_strict="silent")
            if tunit not in (None, "")
            else None
        )
        columns[str(name)] = ColumnDescription(ndim=ndim, unit=unit)
    return columns


def _check_column(name, column, spec):
    """Errors of one column (a `ColumnDescription`) against its spec (ndim, unit).

    The column ``dtype`` is not checked. A spec without ``unit`` expects a
    dimensionless column; units are compared by equivalence.
    """
    errors = []
    ndim = spec.get("ndim")
    if ndim is not None and column.ndim != ndim:
        errors.append(
            f"{name}: Column ndim incorrect. Expected {ndim}, got {column.ndim}."
        )
    try:
        unit = spec.get("unit")
        expected = u.one if unit is None else u.Unit(unit)
    except ValueError as e:
        return errors + [
            f"{name}: invalid unit {spec.get('unit')!r} in definition: {e}"
        ]
    found = u.one if column.unit is None else column.unit
    if not found.is_equivalent(expected):
        errors.append(
            f"{name}: Column unit incorrect. Expected {expected}, got {column.unit}."
        )
    return errors


class DefinitionValidator:
    """Check a header or a table against a definition ``{name: spec}``.

    The same definition layout serves header keywords and table columns (see
    `gammapy.io.registries`). Both share the presence rules (``required``,
    ``required_if``, ``required_unless``, the conditions being read from the
    header); the per-item checks differ:

    * keywords: ``dtype`` (Python type of the value) and ``allowed``;
    * columns: ``ndim`` and ``unit``.
    """

    def __init__(self, definition):
        self.definition = definition

    @classmethod
    def from_dict(cls, definition):
        return cls(definition)

    def check_header(self, header):
        """Errors of a header (dict) against the definition; empty if compliant."""
        errors = []
        missing = [
            name
            for name, spec in self.definition.items()
            if spec.get("required") and _value(header, name) is None
        ]
        if missing:
            errors.append(f"Missing mandatory keyword(s): {missing}")
        errors += self._check_conditions(
            lambda name: _value(header, name) is not None, header
        )
        for name, spec in self.definition.items():
            value = _value(header, name)
            if value is not None:
                errors += _check_keyword(name, value, spec)
        return errors

    def check_table(self, table):
        """Errors of an in-memory table's columns; empty if compliant."""
        return self.check_columns(columns_from_table(table), table.meta)

    def check_columns(self, columns, header=None):
        """Errors of column descriptions ``{name: ColumnDescription}``.

        Build them with `columns_from_table` or `columns_from_header`.
        ``header`` gives the values of the ``required_if`` / ``required_unless``
        conditions.
        """
        errors = []
        header = header or {}
        required = [n for n, s in self.definition.items() if s.get("required")]
        optional = [n for n, s in self.definition.items() if not s.get("required")]
        missing = [name for name in required if name not in columns]
        if missing:
            errors.append(f"Missing mandatory column(s): {missing}")
        for name in required + optional:
            if name in columns:
                errors += _check_column(name, columns[name], self.definition[name])
        errors += self._check_conditions(lambda name: name in columns, header)
        return errors

    def _check_conditions(self, present, header):
        """``required_if`` / ``required_unless``, grouped by condition."""
        conditional = {}
        for name, spec in self.definition.items():
            for rule, negate in (("required_if", False), ("required_unless", True)):
                condition = spec.get(rule)
                if not condition:
                    continue
                holds = all(_value(header, k) == v for k, v in condition.items())
                if holds != negate and not present(name):
                    desc = ", ".join(f"{k}={_value(header, k)!r}" for k in condition)
                    conditional.setdefault(desc, []).append(name)
        return [f"{names} required when {desc}" for desc, names in conditional.items()]

    def with_defaults(self, header):
        """Copy of ``header`` with absent keywords set to their ``default`` (defaults first)."""
        out = {
            name: spec["default"]
            for name, spec in self.definition.items()
            if "default" in spec and _value(header, name) is None
        }
        out.update({k: v for k, v in dict(header).items() if k not in out})
        return out


# --------------- FORMAT VALIDATOR ---------------
@dataclass
class ValidationReport:
    """Outcome of validating one HDU against a (format, version).

    ``table_errors`` is None when no table was checked (header-only HDU).
    """

    hdu: Optional[str]
    format: str
    version: str
    header_errors: list = field(default_factory=list)
    table_errors: Optional[list] = None

    @property
    def header_valid(self):
        return not self.header_errors

    @property
    def table_valid(self):
        return None if self.table_errors is None else not self.table_errors

    @property
    def valid(self):
        return self.header_valid and self.table_valid is not False

    @property
    def errors(self):
        return [f"header: {e}" for e in self.header_errors] + [
            f"table: {e}" for e in (self.table_errors or [])
        ]

    def __str__(self):
        status = "OK" if self.valid else "FAIL"
        lines = [f"{self.hdu} ({self.format} v{self.version}): {status}"]
        lines += [f"  - {e}" for e in self.errors]
        return "\n".join(lines)


class FormatComplianceError(ValueError):
    """Raised by a strict `FormatValidator`; carries the full `ValidationReport`."""

    def __init__(self, report):
        self.report = report
        super().__init__(str(report))


class FormatValidator:
    """Check a table / header against a data format specification.

    Works on the serialized representation (an astropy ``Table`` and its
    ``meta``, a plain header dict, or a FITS HDU), so it can be used
    independently of any reader/writer.

    Parameters
    ----------
    format : str
        Registered data format, e.g. "GADF".
    version : str or None
        Format version to test against. If None, taken from each HDU's
        HDUVERS, falling back to DEFAULT_DATA_FORMAT_VERSION.
    strict : bool
        If True, raise `FormatComplianceError` (after collecting all errors of
        the HDU); otherwise log each error as a warning.
    log_errors : bool
        If False and not strict, errors are only stored in the report (for
        batch validation, where the caller summarises).

    Examples
    --------
    >>> validator = FormatValidator("GADF", "0.3")
    >>> report = validator.validate(table, hdu="AEFF_2D")  # doctest: +SKIP
    >>> report.valid, report.errors  # doctest: +SKIP
    """

    def __init__(
        self,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=False,
        log_errors=True,
    ):
        if format not in DATA_FORMATS_MODELS:
            raise ValueError(
                f"Format {format!r} is not registered; known: {list(DATA_FORMATS_MODELS)}"
            )
        self.format = format
        self.version = version
        self.strict = strict
        self.log_errors = log_errors
        self._models = DATA_FORMATS_MODELS[format]

    # ---- public API ----
    def validate(self, table, hdu=None, meta=None):
        """Validate an in-memory table: its columns and its header.

        The header is ``table.meta`` unless ``meta`` is given. ``hdu`` is the HDU
        class key (e.g. "EVENTS", "AEFF_2D"); if None it is resolved from the
        header with the format's key resolver.
        """
        meta = dict(table.meta if meta is None else meta)
        return self._finalize(self._check(meta, columns_from_table(table), hdu))

    def validate_header(self, header, hdu=None):
        """Validate a FITS header: its keywords and, for a binary table, its columns.

        The columns are described by the ``TTYPEn`` / ``TFORMn`` / ``TDIMn`` /
        ``TUNITn`` keywords, so the table data is not needed. For a header
        without columns (image, primary), only the keywords are checked.
        """
        header = dict(header)
        columns = columns_from_header(header) if "TFIELDS" in header else None
        return self._finalize(self._check(header, columns, hdu))

    def validate_meta(self, meta, hdu=None):
        """Validate header keywords only."""
        return self._finalize(self._check(dict(meta), None, hdu))

    def validate_hdu(self, fits_hdu, hdu=None):
        """Validate a FITS HDU from its header only; the data is not read."""
        if fits_hdu.is_image:
            hdu = hdu or "IMAGE"
        return self.validate_header(fits_hdu.header, hdu=hdu)

    # ---- checks ----
    def _class_key(self, meta, hdu):
        if hdu is not None:
            return hdu.upper()
        return self._models.get("HDU_CLASS_KEY", _hdu_class_key)(meta)

    def _check(self, meta, columns, hdu):
        version = self.version or _value(meta, "HDUVERS") or DEFAULT_DATA_FORMAT_VERSION
        class_key = self._class_key(meta, hdu)
        report = ValidationReport(hdu=class_key, format=self.format, version=version)
        report.header_errors = self._check_header(meta, class_key, version)
        if columns is not None:
            report.table_errors = self._check_columns(columns, meta, class_key, version)
        return report

    def _check_header(self, meta, class_key, version):
        definition = header_definition(self.format, version, class_key)
        errors = DefinitionValidator.from_dict(definition).check_header(meta)
        for check in self._models.get("HEADER_CHECKS", []):
            errors += check(meta, class_key, version)
        return errors

    def _check_columns(self, columns, meta, class_key, version):
        definition = self._models.get("TABLE", {}).get(version, {}).get(class_key)
        if definition is None:
            return [f"No {self.format} v{version} table definition for {class_key!r}"]
        return DefinitionValidator.from_dict(definition).check_columns(columns, meta)

    def _finalize(self, report):
        if report.errors:
            if self.strict:
                raise FormatComplianceError(report)
            if not self.log_errors:
                return report
            for msg in report.errors:
                log.warning("%s: %s", report.hdu, msg)
        return report


def format_class_keys(format):
    """All HDU class keys the format defines (tables or headers, any version)."""
    models = DATA_FORMATS_MODELS.get(format, {})
    keys = set()
    for registry in ("TABLE", "HEADER"):
        for definitions in models.get(registry, {}).values():
            keys |= set(definitions)
    keys.discard("BASE")
    return keys


def validate_hdus(entries, format=DEFAULT_DATA_FORMAT, version=None, checksum=False):
    """Validate many HDUs in batch, opening each file only once.

    No reader/writer or product is built: only the serialized HDUs are checked.

    Parameters
    ----------
    entries : iterable of (filename, hdu, class_key)
        ``hdu`` is the HDU name or index in the file; ``class_key`` the HDU
        class to test against (e.g. "AEFF_2D"), or None to resolve it from the header.
    format, version : str
        Passed to `FormatValidator`; ``version=None`` uses each HDU's HDUVERS.
    checksum : bool
        Verify FITS checksums when opening the files.

    Returns
    -------
    reports : list of `ValidationReport`
        One per entry, same order. Files or HDUs that cannot be read give a
        failed report carrying the IO error.
    """
    entries = list(entries)
    validator = FormatValidator(format, version, strict=False, log_errors=False)
    reports = [None] * len(entries)

    def io_failure(key, hdu, msg):
        return ValidationReport(
            hdu=key or str(hdu),
            format=format,
            version=version or "?",
            header_errors=[msg],
        )

    by_file = {}
    for idx, (filename, hdu, key) in enumerate(entries):
        by_file.setdefault(str(make_path(filename)), []).append((idx, hdu, key))

    for filename, items in by_file.items():
        try:
            # only the headers are used: the data is never loaded
            with fits.open(filename, memmap=True, checksum=checksum) as hdulist:
                for idx, hdu, key in items:
                    try:
                        reports[idx] = validator.validate_hdu(hdulist[hdu], hdu=key)
                    except (KeyError, IndexError, OSError, ValueError, TypeError) as e:
                        reports[idx] = io_failure(
                            key, hdu, f"cannot read HDU {hdu!r}: {e}"
                        )
        except OSError as e:
            for idx, hdu, key in items:
                reports[idx] = io_failure(key, hdu, f"cannot open {filename}: {e}")
    return reports
