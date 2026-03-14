# Owner(s): ["oncall: distributed"]
import copy
import unittest

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.ilp_utils import (
    aggregate_stats,
    get_peak_memory_runtime_baseline,
    ModuleInfo,
    parse_module_info,
)
from torch.distributed._tools.mem_tracker import _ModState, MemTracker
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import SACEstimator, SACStats
from torch.distributed._tools.sac_ilp import (
    get_optimal_checkpointing_policy_per_module,
    sac_milp,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    MI300_ARCH,
    run_tests,
    skipIfRocmArch,
    skipIfTorchDynamo,
    TestCase,
)


class TransformerModel(nn.Module):
    """Sequence-to-sequence model built around ``torch.nn.Transformer``.

    Wraps ``torch.nn.Transformer`` with token-embedding layers so the model
    can accept integer token indices as input, matching the interface used by
    the SAC/ILP estimator helpers.

    ``forward(src)`` internally constructs the target sequence by dropping the
    last token of *src* (teacher-forcing style), so callers pass a single
    ``(batch, seq_len)`` integer tensor — the same API as the old GPT-style
    test model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (bsz, src_seq_len) integer token ids
        # Target is src with the last token dropped (teacher-forcing).
        tgt = src[:, :-1]  # (bsz, tgt_len)
        src_emb = self.src_embedding(src)   # (bsz, src_len, d_model)
        tgt_emb = self.tgt_embedding(tgt)   # (bsz, tgt_len, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=src.device
        )
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)  # (bsz, tgt_len, vocab_size)


class TestSACILP(TestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.cuda.current_device()
        self.estimate_mode = "operator-level-cost-model"
        # Hyper-parameters kept at roughly the same scale as the original
        # GPT-style model (d_model=768, 12 heads, 4 layers) so that the
        # memory-budget thresholds in the three test cases remain sensible.
        self.vocab_size = 8192
        self.d_model = 768
        self.nhead = 12
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dim_feedforward = 3072  # 4 * d_model
        self.dropout = 0.1
        self.max_seq_len = 512
        self.bsz = 8

    def _init_model_input_optimizer(
        self,
    ) -> tuple[nn.Module, torch.optim.Optimizer, torch.Tensor]:
        with torch.device(self.device):
            model = TransformerModel(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        inp = torch.randint(
            0, self.vocab_size, (self.bsz, self.max_seq_len), device=self.device
        )
        return (model, optimizer, inp)

    def _run_and_get_memTracker(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        inp: torch.Tensor,
    ) -> MemTracker:
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optimizer)
        with mem_tracker as mt:
            for iter_idx in range(2):  # running twice to initialize optimizer
                output = model(inp)
                output.sum().backward()
                if iter_idx == 1:
                    last_snapshot = mt.get_tracker_snapshot("current")
                optimizer.step()
                optimizer.zero_grad()
                if iter_idx == 0:
                    mt.reset_mod_stats()
        assert last_snapshot is not None
        for mod_stats in mem_tracker.memory_tracking.values():
            # postprocessing due to the fact that for ModTracker, the post backward hook
            # is not being called for modules whose inputs don't require gradients
            # TODO: fix this in ModTracker and ensure it does not lead to any perf regression
            if _ModState.POST_BW not in mod_stats.snapshots.keys():
                mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
                    copy.deepcopy(last_snapshot)
                )
        return mem_tracker

    def _run_and_get_runtime_estimator(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        inp: torch.Tensor,
    ) -> RuntimeEstimator:
        def _run_one_step() -> None:
            output = model(inp)
            output.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        # Initializing optimizer states and warm-up
        _run_one_step()

        runtime_estimator = RuntimeEstimator()
        with runtime_estimator(estimate_mode_type=self.estimate_mode):
            _run_one_step()  # We use only one iteration for estimation
        return runtime_estimator

    def _run_and_get_sac_estimator(
        self,
        model: nn.Module,
        inp: torch.Tensor,
    ) -> SACEstimator:
        sac_estimator = SACEstimator()
        with sac_estimator(estimate_mode_type=self.estimate_mode):
            loss = model(inp).sum()
        loss.backward()
        return sac_estimator

    def _collect_module_info_with_fake_tensor_mode(self) -> ModuleInfo:
        with FakeTensorMode():
            model, optimizer, inp = self._init_model_input_optimizer()
            mem_tracker = self._run_and_get_memTracker(model, optimizer, inp)
            runtime_estimator = self._run_and_get_runtime_estimator(
                model, optimizer, inp
            )
            sac_estimator = self._run_and_get_sac_estimator(model, inp)
            mod_info = aggregate_stats(
                model,
                mem_tracker,
                runtime_estimator,
                sac_estimator,
                torch.device(self.device),
            )
        return mod_info

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @skipIfRocmArch(MI300_ARCH)
    def test_sac_ilp_case1(self):
        """
        This is a case where the memory budget is either binding or too tight,
        meaning that with some AC, the model can fit into GPU memory.

        With ``torch.nn.Transformer`` the model has
        ``num_encoder_layers`` TransformerEncoderLayer submodules and
        ``num_decoder_layers`` TransformerDecoderLayer submodules.  The ILP
        solver should choose to AC all of them when the budget is tight.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)

        peak_mem, compute_time = get_peak_memory_runtime_baseline(g)
        # Sanity-check: baseline peak memory must be positive and non-trivial.
        self.assertGreater(peak_mem, 0)

        ac_decisions, recomputation_time, _ = sac_milp(
            g, memory_budget=1.6, world_size=4
        )

        # The ILP should decide to apply AC to every transformer layer.
        # With nn.Transformer the layer FQNs follow the pattern:
        #   TransformerModel.transformer.encoder.layers.<i>  (encoder layers)
        #   TransformerModel.transformer.decoder.layers.<i>  (decoder layers)
        n_enc = self.num_encoder_layers
        n_dec = self.num_decoder_layers
        expected_modules = {
            f"TransformerModel.transformer.encoder.layers.{i}"
            for i in range(n_enc)
        } | {
            f"TransformerModel.transformer.decoder.layers.{i}"
            for i in range(n_dec)
        }
        modules_to_ac = set(ac_decisions.keys())
        self.assertEqual(modules_to_ac, expected_modules)

        # All discard ratios must be valid probabilities in (0, 1].
        sorted_discard_ratio = sorted(ac_decisions.values())
        self.assertTrue(
            all(0.0 < r <= 1.0 for r in sorted_discard_ratio),
            f"Unexpected discard ratios: {sorted_discard_ratio}",
        )

        # Recomputation overhead should be positive (AC is active) but still
        # a small fraction of total compute time.
        self.assertGreater(recomputation_time, 0)
        self.assertLess(recomputation_time / compute_time, 0.5)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case2(self):
        """
        This is a case where the memory budget is not binding, meaning that no
        AC is needed to fit the model into memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)
        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=2.4, world_size=4
        )
        self.assertDictEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertGreater(peak_mem, 1)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case3(self):
        """
        This is a case where the memory budget is too tight, meaning that even with
        aggressive AC, the model cannot fit into memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)
        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=0.8, world_size=4
        )
        self.assertEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertEqual(peak_mem, -1)


class TestOptimalCheckpointingPolicy(TestCase):
    # tests are adapted from tests in xformers
    # https://github.com/facebookresearch/xformers/blob/c6c0ac31f1b08542a0bc27278c6ed10f825f6963/tests/test_checkpoint.py#L222
    def setUp(self):
        super().setUp()
        data = [
            ("aten.copy_", 5, 0),
            ("aten.add", 5, 100),
            ("aten.div", 8, 100),
            ("aten.mm", 15, 120),
            ("aten.native_dropout", 15, 0),
            ("aten.linear", 9, 100),
            ("aten.t", 1, 0),
            ("aten.relu_", 5, 0),
        ]
        self.sac_stats = SACStats(
            func_names=[x[0] for x in data],
            runtimes=[x[1] for x in data],
            memory=[x[2] for x in data],
            view_like_ops=[6],
            rand_ops=[4],
            saved_autograd_ops=[],  # not needed for SAC decisions
            inplace_ops=[(0, 0), (7, 5)],
            force_store_random=False,
        )

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_get_optimial_checkpointing_policy_per_module(self):
        for memory_budget, optimal_soln in [
            (0, [1, 0, 0, 0, 1, 0, 0, 0]),
            (100 / 420, [1, 0, 0, 0, 1, 1, 0, 1]),
            (120 / 420, [1, 0, 0, 1, 1, 0, 0, 0]),
            (200 / 420, [1, 0, 1, 0, 1, 1, 0, 1]),
            (220 / 420, [1, 0, 0, 1, 1, 1, 0, 1]),
            (320 / 420, [1, 0, 1, 1, 1, 1, 0, 1]),
            (420 / 420, [1, 1, 1, 1, 1, 1, 0, 1]),
        ]:
            soln = get_optimal_checkpointing_policy_per_module(
                sac_stats=self.sac_stats, memory_budget=memory_budget
            )
            self.assertEqual(optimal_soln, soln)


if __name__ == "__main__":
    run_tests()
