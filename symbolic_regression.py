import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_absolute_error
from collections import defaultdict

df = pd.read_excel("data_gp.xlsx")
X = df[['t', 'mu', 'RA', 'XA', 'XB', 'QA', 'Nd']].values
y = df['VRHE'].values

for pc in np.arange(0.5, 0.95, 0.025):
    for ps in np.arange((0.92-pc), (1-pc), 0.01):
        for parsimony in np.arange(0.0005, 0.0016, 0.0005):
            est_gp = SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01, p_crossover=pc, p_subtree_mutation=ps/3, 
            p_hoist_mutation=ps/3, p_point_mutation=1-pc-ps/3-ps/3, function_set=('add', 'sub', 'mul', 'div', 'sqrt'), 
            parsimony_coefficient=parsimony, tournament_size=20, metric='mean absolute error', const_range=(-1.0, 1.0))

            est_gp.fit(X, y)
            program = str(est_gp._program)
            depth = est_gp._program.depth_
            y_pred = est_gp.predict(X)
            mae = mean_absolute_error(y, y_pred)

            with open("program_length_depth_mae.txt", "a") as f:
                f.write(f"pc: {pc}   ps: {ps}   ph: {ps}   pp: {1-pc-ps-ps}   p_coef: {parsimony}   {program}   length: {length}   depth: {depth}   mae: {mae}\n")


# Input and output file paths
input_file = "program_length_depth_mae.txt"
output_file_depth = "grouped_programs_depth.txt"
output_file_length = "grouped_programs_length.txt"

grouped_lines_length = defaultdict(list)
grouped_lines_depth = defaultdict(list)

# Read and group lines based on length
with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        key = parts[-5]
        grouped_lines_length[key].append(line)

# Read and group lines based on depth
with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        key = parts[-3]
        grouped_lines_depth[key].append(line)

# Sort lines based on the MAE value in each group
def extract_last_number(line):
    return float(line.strip().split()[-1])  # convert last token to float

for key in sorted(grouped_lines_length):
    grouped_lines_length[key] = sorted(grouped_lines_length[key], key=extract_last_number)

for key in sorted(grouped_lines_depth):
    grouped_lines_depth[key] = sorted(grouped_lines_depth[key], key=extract_last_number)

# Write grouped lines to output
with open(output_file_length, "w") as f:
    for key in sorted(grouped_lines_length):  # or just use grouped_lines if order doesn't matter
        f.writelines(grouped_lines_length[key])

with open(output_file_depth, "w") as f:
    for key in sorted(grouped_lines_depth):  # or just use grouped_lines if order doesn't matter
        f.writelines(grouped_lines_depth[key])
