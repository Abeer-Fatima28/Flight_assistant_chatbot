import json
from jinja2 import Template

def render(t: str, **kw)->str:
    return Template(t).render(**kw)

def extract_first_json_block(text: str)->dict:
    s=text.strip()
    if s.startswith('```'):
        s=s.split('\n',1)[1]
        if s.endswith('```'):
            s=s.rsplit('\n',1)[0]
    a,b=s.find('{'), s.rfind('}')
    return json.loads(s[a:b+1]) if a!=-1 and b!=-1 and b>a else json.loads(s)
