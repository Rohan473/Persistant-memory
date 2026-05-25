from memory_layer.mcp import handle_jsonrpc, MCPToolRegistry
import json

# Test list tools
req = {'jsonrpc': '2.0', 'method': 'tools/list', 'id': 1}
resp = handle_jsonrpc(req)
print('Tools available:', len(resp['result']['tools']))
for t in resp['result']['tools']:
    print(f'  - {t["name"]}: {t["description"]}')

# Test search
print('\n--- Test search ---')
req2 = {
    'jsonrpc': '2.0',
    'method': 'tools/call',
    'params': {'name': 'search_research_memory', 'query': 'fundamental value', 'k': 3},
    'id': 2
}
resp2 = handle_jsonrpc(req2)
if 'result' in resp2:
    print('Results:', resp2['result']['count'])
    for r in resp2['result']['results'][:2]:
        print(f'  - {r["name"]} ({r["type"]}) score={r["score"]}')