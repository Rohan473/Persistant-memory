from memory_layer.mcp import MCPToolRegistry, handle_jsonrpc

# Test 1: List tools
tools = MCPToolRegistry.list_tools()
print('Tools available:', len(tools))
for t in tools:
    print(f'  - {t["name"]}')

print('\n--- Test research_state ---')
# Test 2: Research state compiler
req = {
    'jsonrpc': '2.0',
    'method': 'tools/call',
    'params': {
        'name': 'compile_research_state',
        'query': 'momentum alphas',
        'k': 3,
        'budget_tokens': 1000
    },
    'id': 1
}
resp = handle_jsonrpc(req)
if 'result' in resp:
    rs = resp['result']
    print('Query:', rs.get('query'))
    print('Active concepts:', rs.get('research_state', {}).get('active_concepts', [])[:5])
    print('Related failures:', rs.get('research_state', {}).get('related_failures', [])[:5])
    print('Recommendations:', rs.get('research_state', {}).get('recommendations', [])[:2])
    print('Tokens used:', rs.get('tokens_used'))
else:
    print('Error:', resp.get('error'))

print('\n--- Test trace stats endpoint ---')
from memory_layer.trace import logger
stats = logger.get_stats()
print('Stats:', stats)