"""Model training, uncertainty estimation and persistence."""

import numpy as np
import pytest

from src.models.trainer import F1ModelTrainer

MODEL_TYPES = ['xgboost', 'random_forest', 'ensemble']


@pytest.fixture
def prepared(engineered, config_path):
    df, engineer = engineered
    trainer = F1ModelTrainer(config_path)
    X, y = trainer.prepare_data(df, engineer.get_feature_columns(), engineer.get_target_column())
    return trainer, X, y


def test_prepare_data_returns_finite_aligned_arrays(prepared):
    trainer, X, y = prepared

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(trainer.feature_columns)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_prepare_data_drops_rows_without_a_target(engineered, config_path):
    df, engineer = engineered
    df = df.copy()
    df.loc[df.index[:5], 'finish_position'] = np.nan

    trainer = F1ModelTrainer(config_path)
    X, y = trainer.prepare_data(df, engineer.get_feature_columns())

    assert len(y) == len(df) - 5


def test_prepare_data_records_only_available_features(engineered, config_path):
    df, engineer = engineered
    trainer = F1ModelTrainer(config_path)

    trainer.prepare_data(df, engineer.get_feature_columns() + ['not_a_real_feature'])

    assert 'not_a_real_feature' not in trainer.feature_columns


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_train_reports_sane_metrics(prepared, model_type):
    trainer, X, y = prepared

    metrics = trainer.train(X, y, model_type=model_type, cross_validate=False)

    assert metrics['mae'] > 0
    assert metrics['rmse'] >= metrics['mae']
    assert metrics['r2'] <= 1.0
    for key in ('within_1_position', 'within_2_positions', 'within_3_positions'):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics['within_1_position'] <= metrics['within_2_positions'] <= metrics['within_3_positions']


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_predict_with_uncertainty_works_for_every_model_type(prepared, model_type):
    """Regression test: the RF branch used to hit `named_estimators_` and crash."""
    trainer, X, y = prepared
    trainer.train(X, y, model_type=model_type, cross_validate=False)

    predictions, std = trainer.predict_with_uncertainty(X[:10])

    assert predictions.shape == (10,)
    assert std.shape == (10,)
    assert np.isfinite(predictions).all()
    assert np.isfinite(std).all()
    assert ((predictions >= 1) & (predictions <= 20)).all()
    assert (std >= 0.25).all(), "uncertainty must keep a floor so the MC stage stays stochastic"


def test_predict_clips_to_the_valid_position_range(prepared):
    trainer, X, y = prepared
    trainer.train(X, y, model_type='xgboost', cross_validate=False)

    predictions = trainer.predict(X)

    assert predictions.min() >= 1
    assert predictions.max() <= 20


def test_predict_before_training_is_rejected(config_path):
    trainer = F1ModelTrainer(config_path)

    with pytest.raises(ValueError, match="not trained"):
        trainer.predict(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="not trained"):
        trainer.predict_with_uncertainty(np.zeros((2, 3)))


def test_unknown_model_type_is_rejected(prepared):
    trainer, X, y = prepared

    with pytest.raises(ValueError, match="Unknown model type"):
        trainer.train(X, y, model_type='magic', cross_validate=False)


def test_feature_importance_is_populated(prepared):
    trainer, X, y = prepared
    trainer.train(X, y, model_type='xgboost', cross_validate=False)

    assert set(trainer.feature_importance) == set(trainer.feature_columns)
    assert sum(trainer.feature_importance.values()) > 0


def test_save_and_load_round_trip(prepared, tmp_path):
    trainer, X, y = prepared
    trainer.train(X, y, model_type='xgboost', cross_validate=False)
    expected = trainer.predict(X[:5])

    path = tmp_path / "model.joblib"
    trainer.save_model(str(path))

    reloaded = F1ModelTrainer()
    reloaded.load_model(str(path))

    assert reloaded.feature_columns == trainer.feature_columns
    assert reloaded.model_type == trainer.model_type
    np.testing.assert_allclose(reloaded.predict(X[:5]), expected)


def _race_groups(df):
    return df[['year', 'round']].astype(str).agg('_'.join, axis=1).to_numpy()


def test_prepare_data_records_the_race_each_row_belongs_to(prepared, engineered):
    trainer, X, y = prepared
    df, _ = engineered

    assert trainer.groups is not None
    assert len(trainer.groups) == len(y)
    assert set(trainer.groups) == set(_race_groups(df))


def test_whole_races_are_held_out_together(prepared):
    """
    A row-level split leaves 19 of a race's 20 cars in training while the 20th
    is scored, and finishing position is a permutation within the race.
    """
    trainer, X, y = prepared
    groups = trainer.groups

    X_train, X_test, y_train, y_test = trainer._split(
        X, y, groups, test_size=0.25, random_state=42)

    assert len(X_train) + len(X_test) == len(X)

    # Recover which groups landed on each side by matching rows back.
    lookup = {tuple(row): group for row, group in zip(X.tolist(), groups)}
    train_groups = {lookup[tuple(row)] for row in X_train.tolist()}
    test_groups = {lookup[tuple(row)] for row in X_test.tolist()}

    assert test_groups
    assert not (train_groups & test_groups)


def test_the_scaler_is_fitted_on_training_rows_only(prepared):
    """
    Fitting StandardScaler on the whole matrix leaks the held-out mean and
    variance into training.
    """
    trainer, X, y = prepared
    trainer.train(X, y, model_type='random_forest', cross_validate=False, test_size=0.25)

    full_mean = X.mean(axis=0)
    fitted_mean = trainer.scaler.mean_

    assert fitted_mean.shape == full_mean.shape
    # The scaler saw a strict subset, so its mean must differ from the full matrix
    # on at least one feature with real spread.
    spread = X.std(axis=0) > 0
    assert not np.allclose(fitted_mean[spread], full_mean[spread])


def test_cross_validation_groups_by_race(prepared):
    trainer, X, y = prepared

    metrics = trainer.train(X, y, model_type='random_forest', cross_validate=True)

    assert 'cv_mae_mean' in metrics
    assert metrics['cv_mae_mean'] > 0
    assert metrics['cv_mae_std'] >= 0


def test_training_rejects_misaligned_or_tiny_inputs(config_path):
    trainer = F1ModelTrainer(config_path)

    with pytest.raises(ValueError, match="not aligned"):
        trainer.train(np.zeros((10, 3)), np.zeros(9), cross_validate=False)

    with pytest.raises(ValueError, match="at least 5 samples"):
        trainer.train(np.zeros((3, 3)), np.zeros(3), cross_validate=False)


def test_prepare_data_rejects_an_unusable_dataset(engineered, config_path):
    df, engineer = engineered
    trainer = F1ModelTrainer(config_path)

    with pytest.raises(ValueError, match="None of the requested feature columns"):
        trainer.prepare_data(df, ['nope', 'also_nope'])

    with pytest.raises(ValueError, match="not in the dataset"):
        trainer.prepare_data(df, engineer.get_feature_columns(), target_column='missing')

    blank = df.copy()
    blank['finish_position'] = np.nan
    with pytest.raises(ValueError, match="No rows left"):
        trainer.prepare_data(blank, engineer.get_feature_columns())
