from memory_layer import load_metadata

meta = load_metadata()

# Show a few Alpha examples
alpha_count = 0
for m in meta:
    if m.get('node_type') == 'Alpha' and alpha_count < 3:
        print(f"Name: {m['name']}")
        print(f"Summary: {m['structured_summary']}")
        print(f"Operators: {m.get('operators', [])}")
        print(f"Concepts: {m.get('concepts', [])}")
        print(f"Failures: {m.get('failure_modes', [])}")
        print()
        alpha_count += 1