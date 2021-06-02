import shutil

with open('./data/181250059.txt') as file:
    lines = file.readlines()
for line in lines:
    num, person = line.split()
    src = './data/test/{}'.format(num.rjust(10, '0'))
    dst = './data/results/{}/{}'.format(person, num)
    shutil.copyfile(src, dst)
