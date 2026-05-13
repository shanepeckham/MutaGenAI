"""Quick comparison script — run once then delete."""
import json

with open("browsergym_evolution_log.json") as f:
    data = json.load(f)

e = data["evolved"]
b = data["baseline"]
tok = data["token_optimization"]

print("=" * 65)
print("  NEW RUN — Token Optimization Enabled (GPT-4.1)")
print("=" * 65)
print(f"  Timestamp: {data['timestamp']}")
print(f"  Backend:   {data['backend']} / {data['model']}")
cfg = data["config"]
print(f"  Config:    {cfg['iterations']} gen, pop {cfg['population_size']}, "
      f"{cfg['num_islands']} islands, eval_sample={cfg['eval_sample_size']}")
print()

print("  BASELINE:")
print(f"    Score:           {b['score']:.2f}%")
print(f"    Intent accuracy: {b['intent_accuracy']:.1f}%")
print(f"    Per-category:    {b.get('per_category', {})}")
print(f"    Tokens:          {tok['baseline_tokens']}")
print()

print("  EVOLVED:")
print(f"    Score:           {e['score']:.2f}%")
print(f"    Intent accuracy: {e['intent_accuracy']:.1f}%")
print(f"    Per-category:    {e.get('per_category', {})}")
print(f"    Tokens:          {tok['evolved_tokens']}")
print()

print("  DELTA:")
print(f"    Score:  {e['score'] - b['score']:+.2f}%")
print(f"    Intent: {e['intent_accuracy'] - b['intent_accuracy']:+.1f}%")
print(f"    Tokens: {tok['token_delta']:+d} ({tok['evolved_tokens']} vs {tok['baseline_tokens']})")
print()

print("  TOKEN OPTIMIZATION CONFIG:")
for k, v in tok.items():
    print(f"    {k}: {v}")
print()

scoring = data.get("scoring", {})
if scoring:
    print("  SCORING:")
    for k, v in scoring.items():
        print(f"    {k}: {v}")
    print()

print(f"  Wall time: {data.get('wall_time', '?')}")
print()

lb = data.get("leaderboard", [])[:8]
print("  LEADERBOARD (top 8):")
print(f"    {'#':<3} {'Score':<7} {'Gen':<4} {'Operation':<14} {'Tokens'}")
print(f"    {'---'} {'------'} {'---'} {'-------------'} {'------'}")
for i, x in enumerate(lb):
    pt = x.get("prompt_tokens", "?")
    print(f"    {i+1:<3} {x['score']:<7.2f} {x.get('generation','?'):<4} "
          f"{x.get('operation','?'):<14} {pt}")
print()

print("  EVOLVED PROMPT:")
print("  " + "-" * 60)
prompt = e.get("prompt", "")
for line in prompt.split("\n")[:15]:
    print(f"    {line}")
if prompt.count("\n") > 15:
    print(f"    ... ({prompt.count(chr(10)) + 1} total lines)")
print("  " + "-" * 60)
