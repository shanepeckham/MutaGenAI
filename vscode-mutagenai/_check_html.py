import re

with open("src/evolvePanel.ts") as f:
    content = f.read()

start = content.find('private getHtml(): string {')
if start == -1:
    print("getHtml not found")
    exit()

tl_start = content.index('`', start)
i = tl_start + 1
while i < len(content):
    ch = content[i]
    if ch == '`':
        break
    if ch == '\\' and i + 1 < len(content):
        i += 2
        continue
    i += 1
tl_end = i
html = content[tl_start + 1 : tl_end]
print(f"HTML template: {len(html)} chars, {len(html.splitlines())} lines")

# Find ${...} interpolations
interps = list(re.finditer(r'\$\{[^}]*\}', html))
print(f"Template interpolations: {len(interps)}")
for m in interps:
    pos = m.start()
    line = html[:pos].count('\n') + 1
    print(f"  Line {line}: {m.group()}")

# Check for backticks
backticks = [i for i, c in enumerate(html) if c == '`']
print(f"Stray backticks: {len(backticks)}")
for pos in backticks:
    line = html[:pos].count('\n') + 1
    ctx = html[max(0,pos-20):pos+20]
    print(f"  Line {line}: ...{repr(ctx)}...")

# Check for non-ASCII / control chars
for i, c in enumerate(html):
    o = ord(c)
    if o > 127 or (o < 32 and c not in '\n\r\t'):
        line = html[:i].count('\n') + 1
        print(f"  Non-ASCII/control at line {line}, offset {i}: U+{o:04X} = {repr(c)}")
