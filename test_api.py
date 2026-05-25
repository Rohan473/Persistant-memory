import requests

# Test search
r = requests.post('http://localhost:8000/mcp', json={
    'jsonrpc': '2.0',
    'method': 'tools/call',
    'params': {'name': 'search_research_memory', 'query': 'fundamental value alphas', 'k': 3},
    'id': 2
})
result = r.json().get('result', {})
print('Results:', result.get('count'))
for x in result.get('results', [])[:3]:
    print(f"  {x['name']} ({x['type']}) score={x['score']}")

print()

# Test context
r2 = requests.post('http://localhost:8000/mcp', json={
    'jsonrpc': '2.0',
    'method': 'tools/call',
    'params': {'name': 'get_research_context', 'query': 'momentum', 'k': 2, 'format': 'compact'},
    'id': 3
})
result2 = r2.json().get('result', {})
print('Context tokens:', result2.get('tokens'))
print('Context preview:', result2.get('context', '')[:200])