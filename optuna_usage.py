import optuna


fs = [

]

names = [
    ''
]


for f, name in zip(fs, names):
    def objective(trial):
        x = trial.suggest_float('x', -100, 100)
        return (x - 2) ** 2


    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    study.optimize(objective, n_trials=1000)
    print(study.best_params)

