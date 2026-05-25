try:
    from memory_layer.rerank import compile_research_state
    print('Import OK')
    # Try calling it
    result = compile_research_state("test", [], 1000)
    print('Function OK:', result)
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')

# Also test rerank_results
try:
    from memory_layer.rerank import rerank_results
    print('rerank_results OK')
    result = rerank_results([], None, "")
    print('rerank_results result:', result)
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')