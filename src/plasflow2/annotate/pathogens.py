"""Pathogenic bacteria detection from taxonomy assignments.

Identifies contigs from known human-pathogenic species and classifies them
by threat level using WHO/CDC/ESKAPE priority lists.

Usage
-----
    from plasflow2.annotate.pathogens import detect_pathogens, PathogenResult

    # After assign_taxonomy() returns a dict[contig_id -> TaxResult]:
    pathogens = detect_pathogens(taxonomy_by_contig)
    # pathogens: dict[contig_id -> PathogenResult]  (only pathogenic contigs)

Threat levels
-------------
* ``critical``  — WHO critical-priority or ESKAPE pathogen; extremely high
                  clinical relevance (MDR Klebsiella, Acinetobacter, MRSA, …)
* ``high``      — WHO high-priority or CDC urgent/serious threat
                  (Enterococcus, Salmonella, Campylobacter, Streptococcus, …)
* ``medium``    — WHO medium-priority or regionally important
                  (Staphylococcus, Haemophilus, Moraxella, Clostridium, …)

Categories
----------
* ``ESKAPE``    — Enterococcus, Staphylococcus, Klebsiella, Acinetobacter,
                  Pseudomonas, Enterobacter, Escherichia
* ``WHO``       — 2024 WHO Bacterial Priority Pathogens List
* ``CDC``       — CDC 2019 AR Threat Report (urgent/serious tier)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

# ---------------------------------------------------------------------------
# Pathogen reference database
# ---------------------------------------------------------------------------

# Each entry: (genus_match, species_keyword_or_None, threat_level, category, note)
# genus_match is matched case-insensitively against TaxResult.taxon at any rank.
# species_keyword refines the match when genus alone isn't sufficient.

_PATHOGEN_DB: list[tuple[str, str | None, str, str, str]] = [
    # ── ESKAPE + WHO Critical ────────────────────────────────────────────────
    ("Enterococcus", "faecium", "critical", "ESKAPE/WHO", "VRE — vancomycin-resistant"),
    ("Enterococcus", "faecalis", "critical", "ESKAPE/WHO", "VRE — vancomycin-resistant"),
    ("Enterococcus", None, "high", "ESKAPE", "Enterococcus spp."),
    ("Staphylococcus", "aureus", "critical", "ESKAPE/WHO", "MRSA — methicillin-resistant"),
    ("Staphylococcus", "epidermidis", "high", "ESKAPE", "CoNS — coagulase-negative"),
    ("Staphylococcus", None, "medium", "ESKAPE", "Staphylococcus spp."),
    ("Klebsiella", "pneumoniae", "critical", "ESKAPE/WHO", "CRE / carbapenem-resistant"),
    ("Klebsiella", "oxytoca", "high", "ESKAPE/WHO", "Extended-spectrum beta-lactamase"),
    ("Klebsiella", None, "high", "ESKAPE/WHO", "Klebsiella spp."),
    ("Acinetobacter", "baumannii", "critical", "ESKAPE/WHO", "Pandrug-resistant nosocomial"),
    ("Acinetobacter", None, "high", "ESKAPE/WHO", "Acinetobacter spp."),
    ("Pseudomonas", "aeruginosa", "critical", "ESKAPE/WHO", "Carbapenem-resistant P. aeruginosa"),
    ("Pseudomonas", None, "high", "ESKAPE", "Pseudomonas spp."),
    ("Enterobacter", "cloacae", "critical", "ESKAPE/WHO", "Carbapenem-resistant Enterobacter"),
    ("Enterobacter", "hormaechei", "high", "ESKAPE/WHO", "Nosocomial"),
    ("Enterobacter", None, "high", "ESKAPE", "Enterobacter spp."),
    ("Escherichia", "coli", "critical", "ESKAPE/WHO", "ESBL/CRE — O157:H7 STEC"),
    ("Escherichia", None, "high", "ESKAPE", "Escherichia spp."),
    # ── WHO Critical (non-ESKAPE) ────────────────────────────────────────────
    ("Mycobacterium", "tuberculosis", "critical", "WHO", "TB — extensively drug-resistant"),
    ("Mycobacterium", "abscessus", "critical", "WHO", "NTM — intrinsically resistant"),
    ("Mycobacterium", None, "high", "WHO", "Mycobacterium spp."),
    ("Serratia", "marcescens", "critical", "WHO", "Nosocomial, intrinsic resistance"),
    ("Proteus", "mirabilis", "high", "WHO", "ESBL producer"),
    ("Morganella", "morganii", "high", "WHO", "AmpC beta-lactamase"),
    ("Providencia", None, "high", "WHO", "MDR Enterobacteriaceae"),
    ("Citrobacter", "freundii", "high", "WHO", "Carbapenemase producer"),
    ("Citrobacter", None, "medium", "WHO", "Citrobacter spp."),
    # ── WHO High priority ────────────────────────────────────────────────────
    ("Salmonella", "typhi", "critical", "WHO/CDC", "Typhoid — fluoroquinolone-resistant"),
    ("Salmonella", "enterica", "high", "WHO/CDC", "Non-typhoidal salmonella"),
    ("Salmonella", None, "high", "WHO/CDC", "Salmonella spp."),
    ("Shigella", None, "high", "WHO/CDC", "Drug-resistant dysentery"),
    ("Campylobacter", "jejuni", "high", "WHO/CDC", "Fluoroquinolone-resistant"),
    ("Campylobacter", "coli", "high", "WHO/CDC", "Erythromycin-resistant"),
    ("Campylobacter", None, "medium", "WHO/CDC", "Campylobacter spp."),
    ("Haemophilus", "influenzae", "high", "WHO", "Ampicillin-resistant"),
    ("Streptococcus", "pneumoniae", "critical", "WHO/CDC", "Drug-resistant S. pneumoniae"),
    ("Streptococcus", "pyogenes", "high", "WHO", "GAS — iGAS"),
    ("Streptococcus", None, "medium", "WHO", "Streptococcus spp."),
    ("Neisseria", "gonorrhoeae", "critical", "WHO/CDC", "Drug-resistant gonorrhoea"),
    ("Neisseria", "meningitidis", "high", "WHO", "Meningococcal disease"),
    # ── CDC urgent/serious ───────────────────────────────────────────────────
    ("Clostridioides", "difficile", "critical", "CDC", "CDI — hypervirulent strains"),
    ("Clostridium", "difficile", "critical", "CDC", "CDI legacy name"),
    ("Clostridium", None, "medium", "CDC", "Clostridium spp."),
    ("Helicobacter", "pylori", "critical", "WHO", "Clarithromycin-resistant"),
    ("Helicobacter", None, "medium", "WHO", "Helicobacter spp."),
    ("Vibrio", "cholerae", "high", "WHO", "Cholera"),
    ("Vibrio", None, "medium", "WHO", "Vibrio spp."),
    ("Yersinia", "pestis", "critical", "CDC", "Plague"),
    ("Yersinia", "enterocolitica", "high", "WHO", "Yersiniosis"),
    # ── Medium / environmental ──────────────────────────────────────────────
    ("Burkholderia", "pseudomallei", "critical", "CDC", "Melioidosis — select agent"),
    ("Burkholderia", "mallei", "critical", "CDC", "Glanders — select agent"),
    ("Burkholderia", "cepacia", "high", "WHO", "CF pathogen; MDR"),
    ("Burkholderia", None, "medium", "WHO", "Burkholderia spp."),
    ("Stenotrophomonas", "maltophilia", "high", "WHO", "Intrinsic MDR; nosocomial"),
    ("Listeria", "monocytogenes", "high", "WHO", "Listeriosis; immunocompromised"),
    ("Brucella", None, "high", "WHO", "Brucellosis — occupational"),
    ("Legionella", "pneumophila", "high", "WHO", "Legionnaire's — environmental"),
    ("Francisella", "tularensis", "critical", "CDC", "Tularemia — select agent"),
    ("Coxiella", "burnetii", "high", "CDC", "Q fever — environmental"),
    ("Leptospira", None, "medium", "WHO", "Leptospirosis — waterborne"),
    # ── Wastewater-relevant ──────────────────────────────────────────────────
    ("Arcobacter", None, "medium", "WHO", "Emerging waterborne pathogen"),
    ("Aliarcobacter", None, "medium", "WHO", "Emerging waterborne pathogen"),
    ("Arcobacter", "butzleri", "high", "WHO", "Diarrheal disease"),
    ("Aeromonas", "hydrophila", "high", "WHO", "Waterborne; wound infections"),
    ("Aeromonas", None, "medium", "WHO", "Aeromonas spp. — waterborne"),
    ("Mycobacterium", "avium", "high", "WHO", "NTM — waterborne"),
]


# ── Build fast genus-level lookup ────────────────────────────────────────────

_GENUS_INDEX: dict[str, list[tuple[str | None, str, str, str]]] = {}
for _genus, _sp, _level, _cat, _note in _PATHOGEN_DB:
    _GENUS_INDEX.setdefault(_genus.lower(), []).append((_sp, _level, _cat, _note))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PathogenResult:
    """Pathogen annotation for a single contig."""

    contig_id: str
    genus: str
    species: str  # full species name, e.g. "Klebsiella pneumoniae"
    threat_level: str  # "critical" | "high" | "medium"
    category: str  # e.g. "ESKAPE/WHO"
    note: str  # short clinical note

    @property
    def is_eskape(self) -> bool:
        return "ESKAPE" in self.category

    @property
    def is_who_critical(self) -> bool:
        return "WHO" in self.category and self.threat_level == "critical"

    @property
    def badge_class(self) -> str:
        """CSS class for the HTML badge."""
        return {"critical": "besk", "high": "bwho", "medium": "bcard"}.get(
            self.threat_level, "bcard"
        )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _genus_from_lineage(lineage: str) -> str:
    """Extract genus name from a GTDB-style lineage string (g__ prefix)."""
    for part in lineage.split(";"):
        part = part.strip()
        if part.startswith("g__"):
            return part[3:].strip()
    return ""


def _species_from_lineage(lineage: str) -> str:
    """Extract species name from a GTDB-style lineage string (s__ prefix)."""
    for part in lineage.split(";"):
        part = part.strip()
        if part.startswith("s__"):
            return part[3:].strip()
    return ""


def _match_pathogen(genus: str, species: str) -> tuple[str, str, str] | None:
    """Return (threat_level, category, note) for the best matching entry, or None."""
    candidates = _GENUS_INDEX.get(genus.lower())
    if not candidates:
        return None

    # Sort: specific species matches first
    best: tuple[str | None, str, str, str] | None = None
    for sp_kw, level, cat, note in candidates:
        if sp_kw is None:
            # genus-level fallback — use only if no species match found
            if best is None:
                best = (sp_kw, level, cat, note)
        else:
            if species and sp_kw.lower() in species.lower():
                best = (sp_kw, level, cat, note)
                break  # specific match wins

    if best is None:
        return None
    return best[1], best[2], best[3]


def detect_pathogens(
    taxonomy: dict,  # dict[contig_id → TaxResult]
    min_rank: str = "genus",
) -> dict[str, PathogenResult]:
    """Identify contigs from pathogenic organisms.

    Args:
        taxonomy: Output of ``assign_taxonomy()`` — {contig_id: TaxResult}.
        min_rank: Minimum taxonomic rank required for a positive call.
                  Contigs only classified at domain/phylum level are skipped.

    Returns:
        {contig_id: PathogenResult} for all pathogenic contigs.
        Non-pathogenic / unclassified contigs are not included.
    """
    _SKIP_RANKS = {"domain", "phylum", "class", "order", "unclassified", ""}
    results: dict[str, PathogenResult] = {}

    for contig_id, tax in taxonomy.items():
        rank = getattr(tax, "rank", "") or ""
        if rank in _SKIP_RANKS:
            continue
        lineage = getattr(tax, "lineage", "") or ""
        taxon = getattr(tax, "taxon", "") or ""

        genus = _genus_from_lineage(lineage)
        species = _species_from_lineage(lineage)

        if not genus:
            # try extracting from taxon string directly
            genus = taxon.split()[0] if taxon else ""

        if not genus:
            continue

        match = _match_pathogen(genus, species)
        if match is None:
            continue

        threat_level, category, note = match
        display_species = species or genus

        results[contig_id] = PathogenResult(
            contig_id=contig_id,
            genus=genus,
            species=display_species,
            threat_level=threat_level,
            category=category,
            note=note,
        )

    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def pathogen_summary(
    pathogens: dict[str, PathogenResult],
) -> dict:
    """Aggregate pathogen detection results for reporting.

    Returns a dict with keys:
    - total (int): total pathogenic contigs
    - by_level (dict): {threat_level: count}
    - by_species (dict): {species_name: count} sorted by count desc
    - eskape_count (int)
    - who_critical_count (int)
    """
    from collections import Counter

    by_level: Counter[str] = Counter()
    by_species: Counter[str] = Counter()
    eskape = 0
    who_crit = 0

    for pr in pathogens.values():
        by_level[pr.threat_level] += 1
        by_species[pr.species] += 1
        if pr.is_eskape:
            eskape += 1
        if pr.is_who_critical:
            who_crit += 1

    return {
        "total": len(pathogens),
        "by_level": dict(by_level),
        "by_species": dict(by_species.most_common(20)),
        "eskape_count": eskape,
        "who_critical_count": who_crit,
    }


def iter_pathogens_by_threat(
    pathogens: dict[str, PathogenResult],
) -> Iterator[PathogenResult]:
    """Yield PathogenResults sorted critical → high → medium."""
    _ORDER = {"critical": 0, "high": 1, "medium": 2}
    yield from sorted(pathogens.values(), key=lambda r: _ORDER.get(r.threat_level, 9))
