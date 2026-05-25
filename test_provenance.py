print('=== Testing provenance and ontology ===')

# Test provenance
from memory_layer.provenance import add_provenance, get_entity_provenance, provenance_store

p = add_provenance(
    entity_type='concept',
    entity_value='mean_reversion',
    source_type='external_paper',
    source_url='https://arxiv.org/test',
    source_title='Quant Research Paper',
    extraction_confidence=0.85,
    extraction_method='ontology_match',
    matched_patterns=['mean reversion', 'reversal']
)
print(f'Added provenance: {p.trust_score}')

trust = get_entity_provenance('concept', 'mean_reversion')
print(f'Trust metrics: {trust["trust_metrics"]}')

# Test ontology
from memory_layer.ontology import ontology, resolve_to_canonical

# Test resolution
result = resolve_to_canonical('short-term reversal')
print(f'Resolved short-term reversal to: {result}')

result2 = resolve_to_canonical('contrarian')
print(f'Resolved contrarian to: {result2}')

# List concepts
concepts = ontology.list_concepts(category='factor')
print(f'Factor concepts: {len(concepts)}')

# Test knowledge parser with provenance
from memory_layer.knowledge_parser import extract_research_entities
sample = '''
This paper presents a momentum strategy using ts_rank. 
We find that mean reversion occurs at shorter horizons.
The strategy uses group_neutralize for sector adjustment.
'''

result = extract_research_entities(sample, 'https://arxiv.org/test', 'Test Paper')
print(f'\nExtracted concepts: {len(result["concepts"])}')
print(f'Provenance records: {len(result["provenance"])}')

for prov in result['provenance'][:3]:
    print(f'  - {prov["entity_value"]}: trust={prov["extraction_confidence"]}, mapped_to={prov["mapped_to"]}')

print('\n=== All working! ===')