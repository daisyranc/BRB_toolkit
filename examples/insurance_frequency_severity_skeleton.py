"""Frequency-severity workflow skeleton / 频率-强度双 BRB 工作流骨架。"""

from brb import BRBModel, Trainer, evaluate_model, initialize_model_from_data, prepare_tensors


def train_frequency_model(config, train_df, val_df, test_df):
    """Train a frequency BRB with Poisson NLL. / 使用 Poisson NLL 训练频率 BRB。"""
    x_train, y_train, _ = prepare_tensors(train_df, config)
    x_val, y_val, _ = prepare_tensors(val_df, config)
    x_test, y_test, _ = prepare_tensors(test_df, config)

    model = BRBModel(config)
    initialize_model_from_data(model, x_train, y_train, shrinkage=100)
    trainer = Trainer(loss="poisson_nll", optimizer="adam", lr=1e-3)
    trainer.fit(model, x_train, y_train, x_val, y_val)
    return model, evaluate_model(model, x_test, y_test)


def train_severity_model(config, train_df, val_df, test_df, claim_count_column="ClaimNb"):
    """Train a severity BRB with claim-count-weighted Gamma deviance. / 使用赔案次数加权的 Gamma deviance 训练强度 BRB。"""
    # Severity data should contain only rows with ClaimNb > 0.
    # Severity 数据应只保留 ClaimNb > 0 的保单。
    x_train, y_train, w_train = prepare_tensors(train_df, config, claim_count_column)
    x_val, y_val, w_val = prepare_tensors(val_df, config, claim_count_column)
    x_test, y_test, w_test = prepare_tensors(test_df, config, claim_count_column)

    model = BRBModel(config)
    initialize_model_from_data(model, x_train, y_train, sample_weight=w_train, shrinkage=30)
    trainer = Trainer(loss="gamma_deviance", optimizer="adam", lr=1e-3)
    trainer.fit(model, x_train, y_train, x_val, y_val, w_train, w_val)
    return model, evaluate_model(model, x_test, y_test, w_test)
