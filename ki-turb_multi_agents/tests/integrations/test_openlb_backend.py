"""OpenLB backend file-generation and compile-noop tests."""

import integrations
from case_library import make_case
from schemas import SimulationJob
from schemas.simulation_job import JobStatus


def test_openlb_prepare_writes_expected_files(tmp_path):
    backend = integrations.get_backend("openlb")
    case = make_case("hit", "openlb", name="hit_32", resolution=[32, 32, 32])
    job = backend.prepare_case(case, tmp_path / "job", job_id="job_1")

    case_dir = tmp_path / "job"
    assert (case_dir / "case.json").is_file()
    assert (case_dir / "case.xml").is_file()
    assert (case_dir / "provenance.json").is_file()
    assert (case_dir / "run.sh").is_file()
    assert job.status is JobStatus.PREPARED

    xml = (case_dir / "case.xml").read_text(encoding="utf-8")
    assert "<Nx>32</Nx>" in xml
    assert "<Flow>hit</Flow>" in xml
    assert "<Lattice>D3Q19</Lattice>" in xml
    assert "<TurbulenceRegime>les</TurbulenceRegime>" in xml
    assert "<ForcingPattern>random_phase</ForcingPattern>" in xml
    assert "<Collision>Smagorinsky</Collision>" in xml


def test_compile_is_noop_for_ansys_and_palabos():
    for name in ("ansys", "palabos"):
        backend = integrations.get_backend(name)
        job = SimulationJob(job_id="j", backend=name, status=JobStatus.PREPARED)
        out = backend.compile_case(job)
        assert out.status is JobStatus.PREPARED


def test_backend_registry_unchanged():
    assert set(integrations.available_backends()) == {"openlb", "palabos", "ansys"}
