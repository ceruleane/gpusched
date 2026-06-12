import pytest

from gpusched.jobspec import JobSpecError, parse_line, parse_vram


def test_plain_command():
    s = parse_line("python train.py --lr 1e-4", 1, 1)
    assert s.command == "python train.py --lr 1e-4"
    assert s.vram_mib is None and s.n_gpus == 1


def test_blank_and_comment_skipped():
    assert parse_line("   ", 1, 1) is None
    assert parse_line("# a comment", 2, 1) is None


def test_vram_mib():
    assert parse_line("[vram=12000] python x.py", 1, 1).vram_mib == 12000


@pytest.mark.parametrize("val,mib", [("22G", 22528), ("22GiB", 22528), ("1.5g", 1536), ("8000MiB", 8000)])
def test_vram_units(val, mib):
    assert parse_vram(val) == mib


def test_multi_gpu_with_vram():
    s = parse_line("[vram=30G gpus=2] torchrun train.py", 7, 3)
    assert s.vram_mib == 30720 and s.n_gpus == 2 and s.index == 3 and s.lineno == 7


def test_command_with_brackets_later_untouched():
    s = parse_line("python x.py --tags '[a,b]'", 1, 1)
    assert s.vram_mib is None and "[a,b]" in s.command


@pytest.mark.parametrize("line", [
    "[vram=abc] python x.py",
    "[vram=-5] python x.py",
    "[gpus=0] python x.py",
    "[foo=1] python x.py",
    "[vram=8G]",
])
def test_errors_carry_lineno(line):
    with pytest.raises(JobSpecError) as e:
        parse_line(line, 42, 1)
    assert "line 42" in str(e.value)
