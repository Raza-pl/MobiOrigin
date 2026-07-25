from pathlib import Path

from plasflow2.annotate.args import (
    load_amrfinder_metadata,
    parse_amrprot_hits,
)

FAM_HEADER = (
    "#node_id\tparent_node_id\tgene_symbol\thmm_id\thmm_tc1\thmm_tc2\t"
    "blastrule_complete_ident\tblastrule_complete_wp_coverage\t"
    "blastrule_complete_br_coverage\tblastrule_partial_ident\t"
    "blastrule_partial_wp_coverage\tblastrule_partial_br_coverage\t"
    "reportable\ttype\tsubtype\tclass\tsubclass\tfamily_name"
)


def fam_row(
    node: str,
    parent: str,
    symbol: str = "-",
    element_type: str = "",
    drug_class: str = "",
    subclass: str = "",
) -> str:
    fields = [
        node,
        parent,
        symbol,
        "-",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        element_type,
        "",
        drug_class,
        subclass,
        "",
    ]
    return "\t".join(fields)


def modern_title(
    accession: str,
    gene: str,
    family: str,
    subclass: str,
    drug_class: str,
    description: str,
) -> str:
    return "|".join(
        [
            accession,
            "1",
            "1",
            gene,
            family,
            "",
            "1",
            subclass,
            drug_class,
            description,
        ]
    )


def diamond_line(query: str, accession: str, title: str) -> str:
    return "\t".join([query, accession, "99.0", "99.0", "1e-50", title])


def build_metadata(path: Path) -> Path:
    rows = [
        FAM_HEADER,
        fam_row("ALL", ""),
        fam_row("AMR_ROOT", "ALL", element_type="AMR"),
        fam_row("VIR_ROOT", "ALL", element_type="VIRULENCE"),
        fam_row("STRESS_ROOT", "ALL", element_type="STRESS"),
        fam_row(
            "BLATEM",
            "AMR_ROOT",
            "blaTEM",
            drug_class="BETA-LACTAM",
            subclass="BETA-LACTAM",
        ),
        fam_row("STXA", "VIR_ROOT", "stxA2b"),
        fam_row("ARSA", "STRESS_ROOT", "arsA"),
        fam_row(
            "CBLA_AMR",
            "AMR_ROOT",
            "cblA",
            drug_class="BETA-LACTAM",
            subclass="CEPHALOSPORIN",
        ),
        fam_row("CBLA_VIR", "VIR_ROOT", "cblA"),
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_modern_fam_hierarchy_filters_non_amr_hits(tmp_path: Path) -> None:
    metadata = load_amrfinder_metadata(build_metadata(tmp_path / "fam.tsv"))

    titles = [
        modern_title(
            "WP_001.1",
            "blaTEM",
            "blaTEM",
            "BETA-LACTAM",
            "BETA-LACTAM",
            "class_A_beta-lactamase",
        ),
        modern_title(
            "WP_002.1",
            "stxA2b",
            "stxA2b",
            "stxA2b",
            "STX2",
            "Shiga_toxin",
        ),
        modern_title(
            "WP_003.1",
            "arsA",
            "arsA",
            "ARSENITE",
            "ARSENIC",
            "arsenite_efflux_transporter",
        ),
        modern_title(
            "WP_004.1",
            "unknown_toxin",
            "unknown_toxin",
            "",
            "",
            "unknown_toxin",
        ),
    ]

    hits_path = tmp_path / "hits.tsv"
    hits_path.write_text(
        "\n".join(
            diamond_line(f"contig_{index}", f"WP_00{index}.1", title)
            for index, title in enumerate(titles, 1)
        )
        + "\n"
    )

    hits = parse_amrprot_hits(hits_path, metadata)

    assert [hit.gene_name for hit in hits] == ["blaTEM"]
    assert hits[0].drug_class == "beta-lactam"


def test_ambiguous_symbol_requires_amr_header_context(
    tmp_path: Path,
) -> None:
    metadata = load_amrfinder_metadata(build_metadata(tmp_path / "fam.tsv"))

    amr_title = modern_title(
        "WP_010.1",
        "cblA",
        "cblA",
        "CEPHALOSPORIN",
        "BETA-LACTAM",
        "class_A_beta-lactamase",
    )
    virulence_title = modern_title(
        "WP_011.1",
        "cblA",
        "cblA",
        "",
        "",
        "cable_pilus_major_pilin",
    )

    hits_path = tmp_path / "ambiguous.tsv"
    hits_path.write_text(
        diamond_line("amr_contig_1", "WP_010.1", amr_title)
        + "\n"
        + diamond_line(
            "virulence_contig_1",
            "WP_011.1",
            virulence_title,
        )
        + "\n"
    )

    hits = parse_amrprot_hits(hits_path, metadata)

    assert len(hits) == 1
    assert hits[0].gene_name == "cblA"
    assert hits[0].drug_class == "beta-lactam"


def test_legacy_flat_metadata_remains_supported(
    tmp_path: Path,
) -> None:
    fam = tmp_path / "fam.tab"
    fam.write_text("#family_symbol\ttype\tclass\n" "VIM-97\tAMR\tBETA-LACTAM\n")
    metadata = load_amrfinder_metadata(fam)

    hits_path = tmp_path / "legacy.tsv"
    hits_path.write_text(
        "contig_1\tWP_100.1\t99\t99\t1e-50\t"
        "WP_100.1 subclass B1 metallo-beta-lactamase VIM-97 "
        "[Pseudomonas aeruginosa]\n"
    )

    hits = parse_amrprot_hits(hits_path, metadata)

    assert len(hits) == 1
    assert hits[0].gene_name == "VIM-97"
    assert hits[0].drug_class == "beta-lactam"
