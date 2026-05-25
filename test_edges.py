import json

with open('graph/graph.json') as f:
    g = json.load(f)

# Show first few links to understand structure
print("First 5 links:")
for e in g.get('links', [])[:5]:
    print(e)

# Find any link with Operator
print("\n--- Links with Operator ---")
for e in g.get('links', []):
    tgt = e.get('target')
    src = e.get('source')
    if 'Operator' in str(tgt) or 'Operator' in str(src):
        print(f"Source: {src}, Target: {tgt}, Relation: {e.get('relation')}")