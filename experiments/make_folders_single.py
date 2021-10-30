import os

splits = [2, 5, 9, 25]

os.system('mkdir results')
os.chdir('results')
os.system('mkdir results_nosplit && mkdir results_baseline')
for s in splits:
    os.system(f'mkdir results_split{s}')
