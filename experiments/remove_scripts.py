import os

nets = ['convbig', 'convmed', 'ffnn']

files = []
for net in nets:
    files.append(net)
    files.append(net + '_splitting')
    files.append(net + '_compose')
    files.append(net + '_compose_splitting')
    files.append(net + '_classify')

os.system(f'rm {" ".join(files)}')
os.system('rm perturb_baseline perturb_compose_baseline')
