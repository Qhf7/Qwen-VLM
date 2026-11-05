"""CPU checks of trainer scheduling; no torch/model imports or GPU access.

AST extraction executes only the two pure scheduling helpers from the actual
trainer, so loading this test does not import FrozenLanguageBridge/dec_big.
Run: python -m unittest -v test_migration_schedule.py
"""
import ast
import hashlib
import json
from pathlib import Path
import random
import unittest


def load_schedule_helpers():
    source = Path(__file__).with_name("migration_train.py")
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"sample_schedule", "fixed_subset"}]
    if len(selected) != 2:
        raise AssertionError("Trainer must expose sample_schedule and fixed_subset")
    namespace = {"random": random}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["sample_schedule"], namespace["fixed_subset"]


class MigrationScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schedule, subset = load_schedule_helpers()
        cls.schedule = staticmethod(schedule)
        cls.subset = staticmethod(subset)

    def setUp(self):
        self.items = [{"path": f"{split}_{label}_{index}", "category": f"c{index % 12}",
                       "label": label, "split": split}
                      for split, counts in (("train", (100, 20)), ("val", (172, 208)), ("test", (196, 260)))
                      for label, count in enumerate(counts) for index in range(count)]
        self.val_ids = [i for i, item in enumerate(self.items) if item["split"] == "val"]

    def test_full_exposure_budget_is_exactly_balanced_and_train_only(self):
        indices = self.schedule(self.items, seed=42, exposures=1500 * 4 * 1)
        self.assertEqual(len(indices), 6000)
        self.assertEqual(sum(self.items[i]["label"] == 0 for i in indices), 3000)
        self.assertEqual(sum(self.items[i]["label"] == 1 for i in indices), 3000)
        self.assertTrue(all(self.items[i]["split"] == "train" for i in indices))

    def test_seed_and_schedule_hash_are_reproducible(self):
        a = self.schedule(self.items, seed=42, exposures=6000)
        b = self.schedule(self.items, seed=42, exposures=6000)
        c = self.schedule(self.items, seed=43, exposures=6000)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        digest = lambda value: hashlib.sha256(json.dumps(value).encode()).hexdigest()
        self.assertEqual(digest(a), digest(b))
        self.assertNotEqual(digest(a), digest(c))

    def test_helpers_do_not_perturb_global_training_rng(self):
        random.seed(919)
        state = random.getstate()
        self.schedule(self.items, seed=42, exposures=120)
        self.subset(self.val_ids, self.items, 24)
        self.assertEqual(state, random.getstate())

    def test_batched_and_accumulated_schedule_slices_cover_same_exposures(self):
        indices = self.schedule(self.items, seed=42, exposures=6000)
        def consume(batch_size, accum):
            groups = []
            for step in range(1500):
                group = []
                for microstep in range(accum):
                    offset = (step * accum + microstep) * batch_size
                    chunk = indices[offset:offset + batch_size]
                    self.assertEqual(len(chunk), batch_size)
                    group.extend(chunk)
                groups.append(group)
            return groups
        # Exposure ordering equivalence does not assert identical token-mean
        # loss weighting or floating point gradients for two batching schemes.
        self.assertEqual(consume(4, 1), consume(1, 4))
        self.assertEqual([i for group in consume(4, 1) for i in group], indices)

    def test_smoke_24_validation_subset_has_both_labels_and_only_val(self):
        chosen = self.subset(self.val_ids, self.items, 24)
        self.assertEqual(len(chosen), 24)
        self.assertEqual(sum(self.items[i]["label"] for i in chosen), 12)
        self.assertEqual(len(set(chosen)), 24)
        self.assertTrue(all(self.items[i]["split"] == "val" for i in chosen))
        self.assertEqual(chosen, self.subset(self.val_ids, self.items, 24))

    def test_full_validation_and_generate_zero_limit_keep_every_source(self):
        self.assertEqual(self.subset(self.val_ids, self.items, 0), self.val_ids)
        self.assertEqual(self.subset(self.val_ids, self.items, 380), self.val_ids)
        self.assertEqual(len(self.subset(self.val_ids, self.items, 0)), 380)


if __name__ == "__main__":
    unittest.main()
