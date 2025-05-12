import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_excel("data_gp.xlsx")
X = df[['t', 'mu', 'RA', 'XA', 'XB', 'QA', 'Nd']].values
y = df['VRHE'].values

for pc in np.arange(0.5, 0.95, 0.025):
    for ps in np.arange((0.92-pc)/3, (1-pc)/3, 0.01):
        for parsimony in np.arange(0.0005, 0.0016, 0.0005):
            est_gp = SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01, p_crossover=pc, p_subtree_mutation=ps, 
            p_hoist_mutation=ps, p_point_mutation=1-pc-ps-ps, function_set=('add', 'sub', 'mul', 'div', 'sqrt'), 
            parsimony_coefficient=parsimony, tournament_size=20, metric='mean absolute error', const_range=(-1.0, 1.0))

            est_gp.fit(X, y)
            program = str(est_gp._program)
            depth = est_gp._program.depth_
            y_pred = est_gp.predict(X)
            mae = mean_absolute_error(y, y_pred)

            with open("program_depth_mae.txt", "a") as f:
                f.write(f"pc: {pc}   ps: {ps}   ph: {ps}   pp: {1-pc-ps-ps}   p_coef: {parsimony}   {program}   {depth}   {mae}\n")
