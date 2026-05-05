with open('out/evolvePanel.js') as f:
    lines = f.readlines()
for i, line in enumerate(lines[200:952], start=201):
    if '${' in line:
        print(f'{i}: {line.rstrip()[:120]}')
