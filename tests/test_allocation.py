from gpusched.allocation import AllocOptions, Occupant, find_allocation
from gpusched.jobspec import JobSpec
from gpusched.testing import FakeBackend

OPTS = AllocOptions(idle_threshold_mib=200, margin_mib=512)


def spec(i=1, vram=None, gpus=1):
    return JobSpec(index=i, command="x", vram_mib=vram, n_gpus=gpus)


def snap(g):
    return FakeBackend(g).snapshot()


def test_undeclared_needs_idle_gpu():
    s = snap({0: (24000, 5000), 1: (24000, 50)})
    assert find_allocation(spec(), s, [], {0, 1}, OPTS) == [1]


def test_undeclared_blocked_when_nothing_idle():
    s = snap({0: (24000, 5000), 1: (24000, 300)})
    assert find_allocation(spec(), s, [], {0, 1}, OPTS) is None


def test_declared_packs_onto_partially_used_gpu():
    # 24000 total, 10000 externally used -> 14000 free; need 8000+512.
    s = snap({0: (24000, 10000)})
    assert find_allocation(spec(vram=8000), s, [], {0}, OPTS) == [0]


def test_declared_respects_margin():
    s = snap({0: (24000, 16000)})  # 8000 free; 8000 + 512 margin doesn't fit
    assert find_allocation(spec(vram=8000), s, [], {0}, OPTS) is None


def test_launch_race_reservation_blocks_double_booking():
    # GPU empty per nvidia-smi, but a just-launched declared job (actual 0)
    # reserves its full estimate -> only 24000 - 20000 = 4000 headroom left.
    s = snap({0: (24000, 0)})
    occ = [Occupant(gpu_indices=(0,), vram_mib=20000, actual_mib={0: 0})]
    assert find_allocation(spec(vram=8000), s, occ, {0}, OPTS) is None
    assert find_allocation(spec(vram=3000), s, occ, {0}, OPTS) == [0]


def test_reservation_shrinks_as_job_ramps():
    # Same job, now actually using 18000 (reported in used_mib too):
    # pending reservation = max(0, 20000-18000) = 2000; headroom = 24000-18000-2000 = 4000.
    s = snap({0: (24000, 18000)})
    occ = [Occupant(gpu_indices=(0,), vram_mib=20000, actual_mib={0: 18000})]
    assert find_allocation(spec(vram=3000), s, occ, {0}, OPTS) == [0]
    assert find_allocation(spec(vram=4000), s, occ, {0}, OPTS) is None


def test_undeclared_occupant_owns_whole_gpu():
    s = snap({0: (24000, 1000), 1: (24000, 0)})
    occ = [Occupant(gpu_indices=(0,), vram_mib=None, actual_mib={0: 1000})]
    assert find_allocation(spec(vram=1000), s, occ, {0}, OPTS) is None
    assert find_allocation(spec(vram=1000), s, occ, {0, 1}, OPTS) == [1]


def test_exclusive_mode_forbids_colocation_but_allows_external_sharing():
    opts = AllocOptions(idle_threshold_mib=200, margin_mib=512, exclusive=True)
    s = snap({0: (24000, 0), 1: (24000, 10000)})
    occ = [Occupant(gpu_indices=(0,), vram_mib=4000, actual_mib={0: 4000})]
    # gpu0 has plenty of headroom but hosts a scheduler job -> gpu1 despite external use.
    assert find_allocation(spec(vram=8000), s, occ, {0, 1}, opts) == [1]


def test_multi_gpu_requires_per_gpu_headroom():
    s = snap({0: (24000, 0), 1: (24000, 20000), 2: (24000, 100)})
    assert find_allocation(spec(vram=10000, gpus=2), s, [], {0, 1, 2}, OPTS) == [0, 2]
    assert find_allocation(spec(vram=10000, gpus=3), s, [], {0, 1, 2}, OPTS) is None


def test_best_fit_preserves_large_headroom():
    # 6000-MiB job should take the *tighter* gpu1 (8000 free), keeping gpu0
    # (24000 free) open for a later large job.
    s = snap({0: (24000, 0), 1: (24000, 16000)})
    assert find_allocation(spec(vram=6000), s, [], {0, 1}, OPTS) == [1]


def test_allowed_gpus_filter():
    s = snap({0: (24000, 0), 1: (24000, 0)})
    assert find_allocation(spec(), s, [], {1}, OPTS) == [1]
