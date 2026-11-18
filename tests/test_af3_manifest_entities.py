"""AF3 manifests must describe both polymers and non-polymer ligands."""

import io
from pathlib import Path

from atomworks.io import parse
from biotite.structure.io.pdbx import CIFFile

from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY


def test_af3_manifest_retains_ligand_entity_type(tmp_path):
    source = Path(__file__).parent / "data/sam_binder_generated_minimized.cif"
    atoms = parse(source, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
    job = OracleJobInput(atom_array=atoms, example_id="protein_and_sam", out_dir=tmp_path)
    entries, _ = AF3Oracle(seed=0)._build_manifest([job], tmp_path / "raw")

    block = next(iter(CIFFile.read(io.StringIO(entries[0]["cif"])).values()))
    entities = dict(zip(block["entity"]["id"].as_array(str), block["entity"]["type"].as_array(str), strict=True))
    chain_entities = dict(
        zip(
            block["atom_site"]["label_asym_id"].as_array(str),
            block["atom_site"]["label_entity_id"].as_array(str),
            strict=True,
        )
    )
    assert entities[chain_entities["A"]] == "polymer"
    assert entities[chain_entities["B"]] == "non-polymer"
