"""CPU-only relay and patch lifecycle tests; no claim of GPU inference coverage.

Execute the actual patched forward method and pinned upstream mixin/PP relay
helper with deterministic scalar layers. This checks feature ownership/order,
residual reconstruction, stage partitioning, and transport omissions.
"""

import ast
from bisect import bisect_right
import difflib
import importlib.util
from itertools import combinations, islice
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/qwen3next_pp"
spec = importlib.util.spec_from_file_location("patcher", ROOT / "utility/qwen3next_pp_patch.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
ORIGINAL = (FIXTURES / "qwen3_next.py.txt").read_bytes()
PATCHED = patcher.transform(ORIGINAL)


class IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


def model_method(source, name):
    model = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.ClassDef) and n.name == "Qwen3NextModel")
    return next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == name)


def compile_node(node, namespace):
    exec("from __future__ import annotations\n" + ast.unparse(node), namespace)


class PatchLifecycle(unittest.TestCase):
    def test_review_diff_matches_installed_edits(self):
        expected = "".join(difflib.unified_diff(
            ORIGINAL.decode().splitlines(True), PATCHED.decode().splitlines(True),
            fromfile="a/" + patcher.MODEL_FILE, tofile="b/" + patcher.MODEL_FILE))
        self.assertEqual((ROOT / "patches/qwen3next-pp/vllm-0.30.0.patch").read_text(), expected)

    def test_round_trip_and_exact_source(self):
        self.assertEqual(patcher.source_state(ORIGINAL), "upstream")
        self.assertEqual(patcher.source_state(PATCHED), "patched")
        self.assertEqual(patcher.transform(PATCHED, reverse=True), ORIGINAL)
        self.assertEqual(patcher.source_state(ORIGINAL + b"# local change\n"), "unknown")
        self.assertEqual(patcher.source_state(PATCHED + b"# local change\n"), "unknown")

    def test_atomic_idempotent_apply_check_revert(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.py"
            path.write_bytes(ORIGINAL)
            path.chmod(0o640)
            with self.assertRaisesRegex(ValueError, "not installed"):
                patcher.update_source(path, "check")
            for _ in range(2):
                self.assertEqual(patcher.update_source(path, "apply"), "patched")
            self.assertEqual(path.read_bytes(), PATCHED)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(patcher.update_source(path, "check"), "patched")
            path.write_bytes(PATCHED + b"# changed\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                patcher.update_source(path, "revert")
            self.assertEqual(path.read_bytes(), PATCHED + b"# changed\n")
            path.write_bytes(PATCHED)
            for _ in range(2):
                self.assertEqual(patcher.update_source(path, "revert"), "upstream")
            self.assertEqual(path.read_bytes(), ORIGINAL)
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_version_pin(self):
        with patch.object(patcher, "distribution", return_value=SimpleNamespace(version="0.30.1")):
            with self.assertRaisesRegex(ValueError, "exactly"):
                patcher.installed_source()

    def test_opt_in_flag(self):
        model = next(n for n in ast.parse(PATCHED).body
                     if isinstance(n, ast.ClassDef) and n.name == "Qwen3NextModel")
        flag = next(n for n in model.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "supports_aux_hidden_states_over_pp"
                            for t in n.targets))
        for value in (None, "0", "1"):
            environ = {} if value is None else {"TACC_QWEN3NEXT_PP_DFLASH": value}
            namespace = dict(os=SimpleNamespace(environ=environ))
            compile_node(flag, namespace)
            self.assertEqual(namespace["supports_aux_hidden_states_over_pp"], value == "1")

    def test_constructor_scope_guard(self):
        init = model_method(PATCHED, "__init__")
        guard = next(n for n in init.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "self.supports_aux_hidden_states_over_pp")
        def check(**changes):
            values = dict(kind="qwen3_next", tp=1, pp=4, sp=False, method="dflash", v2="1")
            values.update(changes)
            namespace = dict(
                self=SimpleNamespace(supports_aux_hidden_states_over_pp=True),
                config=SimpleNamespace(model_type=values["kind"]),
                parallel_config=SimpleNamespace(tensor_parallel_size=values["tp"],
                    pipeline_parallel_size=values["pp"], use_sequence_parallel_moe=values["sp"]),
                vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method=values["method"])),
                os=SimpleNamespace(environ={"VLLM_USE_V2_MODEL_RUNNER": values["v2"]}),
            )
            compile_node(guard, namespace)
        check()
        for change in [dict(kind="qwen3_5_text"), dict(tp=2), dict(pp=3), dict(sp=True),
                       dict(method="eagle3"), dict(v2="0")]:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                check(**change)


class Relay(unittest.TestCase):
    def setUp(self):
        self.pp = SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
        namespace = dict(get_pp_group=lambda: self.pp, islice=islice,
                         bisect_right=bisect_right, IntermediateTensors=IntermediateTensors)
        exec("from __future__ import annotations\n" + (FIXTURES / "eagle_mixin.py.txt").read_text(), namespace)
        exec("from __future__ import annotations\n" + (FIXTURES / "pp_relay.py.txt").read_text(), namespace)
        self.mixin = namespace["EagleModelMixin"]
        self.relay = namespace["relay_aux_hidden_states"]
        compile_node(model_method(ORIGINAL, "forward"), namespace)
        self.original_forward = namespace["forward"]
        compile_node(model_method(PATCHED, "forward"), namespace)
        self.patched_forward = namespace["forward"]
        module = ModuleType("vllm.distributed.parallel_state")
        module.get_pp_group = lambda: self.pp
        module.model_parallel_is_initialized = lambda: True
        self.modules = patch.dict(sys.modules, {"vllm.distributed.parallel_state": module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def run_pipeline(self, cuts, taps, patched=True, enabled=True, drop_transport=False):
        incoming = None
        for rank, (start, end) in enumerate(zip(cuts, cuts[1:])):
            self.pp.world_size = len(cuts) - 1
            self.pp.is_first_rank = rank == 0
            self.pp.is_last_rank = rank == len(cuts) - 2
            model = self.mixin()
            model.start_layer, model.end_layer = start, end
            model.supports_aux_hidden_states_over_pp = enabled
            model.config = SimpleNamespace(model_type="qwen3_next")
            model.use_sequence_parallel = False
            model.embed_input_ids = lambda x: x
            model.norm = lambda h, r: (h + (r or 0), None)
            def layer(index):
                def forward(positions, hidden_states, residual):
                    if not start <= index < end:
                        raise AssertionError("Executed a layer outside this PP stage")
                    residual = hidden_states + (residual or 0)
                    return residual * (index + 2) + 3, residual
                return forward
            model.layers = [layer(i) for i in range(cuts[-1])]
            model._set_aux_hidden_state_layers(taps)
            forward = self.patched_forward if patched else self.original_forward
            output = forward(model, 2 if rank == 0 else None, SimpleNamespace(shape=(3,)), incoming)
            if not self.pp.is_last_rank:
                # The actual V2 helper forwards upstream slots through middle ranks.
                keys = tuple(f"aux_hidden_states_{i}" for i in range(model._aux_slot_base_cached))
                handler = SimpleNamespace(aux_hidden_state_relay_keys=keys if rank else ())
                output = self.relay(handler, incoming, output)
                if drop_transport:
                    output = IntermediateTensors({k: v for k, v in output.tensors.items()
                                                  if not k.startswith("aux_hidden_states_")})
                incoming = output
        return output

    def test_features_match_unpartitioned_reference(self):
        # Exhaustive triples plus singleton boundary taps and stages with no taps.
        tap_sets = [(), *[(i,) for i in range(13)], *combinations(range(13), 3)]
        for taps in tap_sets:
            reference = self.run_pipeline((0, 12), taps, patched=False, enabled=False)
            for cuts in [(0, 12), (0, 6, 12), (0, 3, 6, 9, 12), (0, 1, 2, 8, 12)]:
                with self.subTest(cuts=cuts, taps=taps):
                    self.assertEqual(self.run_pipeline(cuts, taps), reference)

    def test_disabled_patch_preserves_baseline(self):
        for cuts in [(0, 12), (0, 6, 12), (0, 3, 6, 9, 12)]:
            self.assertEqual(self.run_pipeline(cuts, (), enabled=False),
                             self.run_pipeline(cuts, (), patched=False, enabled=False))
        self.assertEqual(self.run_pipeline((0, 12), (0, 5, 12), enabled=False),
                         self.run_pipeline((0, 12), (0, 5, 12), patched=False, enabled=False))

    def test_missing_transport_is_detected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing aux_hidden_states"):
            self.run_pipeline((0, 6, 12), (1, 9, 12), drop_transport=True)


if __name__ == "__main__":
    unittest.main()
