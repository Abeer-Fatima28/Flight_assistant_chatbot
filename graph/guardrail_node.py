from typing import Dict, Any

class GuardrailNode:
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q=state.get('query','') or ''
        if any(x in q.lower() for x in ['ssn','credit card','passport number']):
            return {'should_block': True, 'response': "Please don't share sensitive numbers. I can help without them."}
        if not q.strip():
            return {'should_block': True, 'response': 'Could you share a bit more detail?'}
        return {'should_block': False}
