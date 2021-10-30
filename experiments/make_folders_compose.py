import os

splits = [2, 3, 5, 9]

os.system('mkdir results_compose')
os.chdir('results_compose')
os.system('mkdir results_compose_nosplit && mkdir results_compose_baseline')
for s in splits:
    os.system(f'mkdir results_compose_split{s}')
