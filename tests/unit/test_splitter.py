"""
test_splitter.py
----------------
Unit tests for execution/preprocessing/splitter.py
"""

import pytest
from collections import Counter

from stratml.execution.preprocessing.splitter import split_dataset
from stratml.execution.schemas import SplitConfig


class TestSplitSizes:
    def test_total_rows_preserved(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified"), "classification")
        total = len(split.X_train) + len(split.X_val) + len(split.X_test)
        assert total == clf_dataset.rows

    def test_test_size_approx(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified", test_size=0.2), "classification")
        assert abs(len(split.X_test) / clf_dataset.rows - 0.2) < 0.05

    def test_val_size_approx(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified", val_size=0.1), "classification")
        assert abs(len(split.X_val) / clf_dataset.rows - 0.1) < 0.05

    def test_X_y_lengths_match(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified"), "classification")
        assert len(split.X_train) == len(split.y_train)
        assert len(split.X_val)   == len(split.y_val)
        assert len(split.X_test)  == len(split.y_test)


class TestStratification:
    def test_class_ratios_preserved_in_train(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified"), "classification")
        counts = Counter(split.y_train)
        # all 3 classes should be roughly equal
        values = list(counts.values())
        assert max(values) - min(values) <= 4  # allow small rounding diff

    def test_class_ratios_preserved_in_val(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified"), "classification")
        classes_in_val = set(split.y_val.unique())
        assert classes_in_val == {"a", "b", "c"}

    def test_regression_uses_random_split(self, reg_dataset):
        # Should not raise — regression uses random (no stratify)
        split = split_dataset(reg_dataset, SplitConfig(method="random"), "regression")
        assert len(split.X_train) > 0


class TestIndexReset:
    def test_train_index_starts_at_zero(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified"), "classification")
        assert split.X_train.index[0] == 0
        assert split.y_train.index[0] == 0

    def test_val_index_starts_at_zero(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified"), "classification")
        assert split.X_val.index[0] == 0

    def test_test_index_starts_at_zero(self, clf_dataset):
        split = split_dataset(clf_dataset, SplitConfig(method="stratified"), "classification")
        assert split.X_test.index[0] == 0


class TestReproducibility:
    def test_same_seed_same_split(self, clf_dataset):
        cfg = SplitConfig(method="stratified", random_seed=42)
        s1 = split_dataset(clf_dataset, cfg, "classification")
        s2 = split_dataset(clf_dataset, cfg, "classification")
        assert list(s1.y_train) == list(s2.y_train)

    def test_different_seed_different_split(self, clf_dataset):
        s1 = split_dataset(clf_dataset, SplitConfig(method="stratified", random_seed=0), "classification")
        s2 = split_dataset(clf_dataset, SplitConfig(method="stratified", random_seed=99), "classification")
        assert list(s1.y_train) != list(s2.y_train)
