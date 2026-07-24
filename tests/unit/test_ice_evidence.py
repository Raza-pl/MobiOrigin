from plasflow2.annotate.ice import ICEHit, filter_coherent_ice_hits


def make_hit(ice_id, function, orf_id):
    return ICEHit(
        contig_id="contig",
        ice_id=ice_id,
        gene_function=function,
        identity=99.0,
        coverage=99.0,
        evalue=1e-50,
        _orf_id=orf_id,
    )


def test_generic_iceberg_arg_is_not_ice_evidence():
    hits = [
        make_hit(
            "2005",
            "class A beta-lactamase; AR, blaTEM-1B",
            "contig_1",
        )
    ]
    assert filter_coherent_ice_hits(hits) == []


def test_integration_and_transfer_confirm_same_ice():
    hits = [
        make_hit("1010", "Integrase", "contig_1"),
        make_hit(
            "1010",
            "conjugal transfer coupling protein",
            "contig_2",
        ),
    ]
    assert filter_coherent_ice_hits(hits) == hits


def test_different_ice_elements_are_not_combined():
    hits = [
        make_hit("1010", "Integrase", "contig_1"),
        make_hit(
            "2020",
            "conjugal transfer coupling protein",
            "contig_2",
        ),
    ]
    assert filter_coherent_ice_hits(hits) == []
