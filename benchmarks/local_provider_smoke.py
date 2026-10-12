#!/usr/bin/env python3
"""Bounded synthetic public-only integrated auxiliary smoke benchmark."""
import argparse, hashlib, json, os, re, sys, tempfile, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--execute', action='store_true'); p.add_argument('--model', default='mcp-local-smoke'); p.add_argument('--repo',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args()
if not a.execute:
    print(json.dumps({'status':'prepared_not_executed','requires':'--execute','calls':7,'max_tokens':512,'timeout_seconds':30})); sys.exit(0)
# Remove ambient authentication, hooks and proxy routes before importing project code.
for key in list(os.environ):
    if any(s in key.upper() for s in ('API_KEY','TOKEN','SECRET','PROXY')) or key in ('SKILL_HUB_LOCAL_ONLY','SKILL_HUB_ROUTER_HAIKU'):
        os.environ.pop(key,None)
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'; os.environ['DO_NOT_TRACK']='1'
root=Path(tempfile.mkdtemp(prefix='skill-hub-public-smoke-',dir='/private/tmp')); os.chdir(root)
sys.path.insert(0,str(a.repo.resolve()/'src'))
from unittest.mock import patch
# Bound all home-based defaults to synthetic scratch via library patch, without changing HOME.
with patch.object(Path,'home',return_value=root):
    from skill_hub import config as cfg
    model='smoke-local::'+a.model
    conf={'llm_provider_registry':[{'name':'smoke-local','kind':'openai_compatible','level':'L1','api_base':'http://127.0.0.1:1234/v1','api_key':{'source':'inline','ref':'local-no-secret'},'enabled':True,'order':1,'models':[{'id':a.model,'complexity':'light'}]}], 'llm_providers':{x:model for x in ('tier_cheap','tier_mid','tier_smart','tier_planner')},'reason_model':model,'searxng_use_llm_summary':True,'searxng_timeout':5,'fanout':{'prompt_model':model,'prompt_timeout':30},'services':{'ollama_daemon':{'enabled':False},'ollama_embed':{'enabled':False},'haiku_router':{'enabled':True}}}
    cfg.CONFIG_PATH=root/'config.json'; cfg.CONFIG_PATH.write_text(json.dumps(conf)); conf=cfg._merge_defaults(conf); cfg.get=lambda k,default=None: conf.get(k,default); cfg.load_config=lambda:conf.copy()
    from skill_hub.llm import litellm_adapter as adapter
    from skill_hub import llm, embeddings, wiki, searxng
    from skill_hub.router import session_memory, haiku_client
    from skill_hub.fanout.prompt_synth import draft_prompt
    from skill_hub.fanout.sources import Issue
    from skill_hub.store import SkillStore
    adapter._emit_llm_event=lambda **kw:None; embeddings.log_llm=lambda *args,**kw:None
    provider=adapter.LitellmProvider(); original_complete=provider.complete
    original_transport=provider._litellm.completion; calls=[]; current=['']
    def transport(**kw):
        if len(calls)>=12: raise RuntimeError('call budget exceeded')
        if kw.get('api_base')!='http://127.0.0.1:1234/v1' or kw.get('model')!='openai/'+a.model:
            raise RuntimeError('unexpected route blocked')
        kw.update(max_tokens=min(int(kw.get('max_tokens',512)),512),timeout=30,num_retries=0)
        t=time.monotonic(); entry={'path':current[0],'model':kw['model'],'max_tokens':kw['max_tokens']}; calls.append(entry)
        try:
            response=original_transport(**kw); choice=response.choices[0]; entry.update(duration_seconds=round(time.monotonic()-t,3),finish_reason=choice.finish_reason,usage=dict(response.usage or {}),raw=choice.message.content or '')
            return response
        except Exception as exc:
            entry.update(duration_seconds=round(time.monotonic()-t,3),error_type=type(exc).__name__); raise
    provider._litellm.completion=transport
    def bounded_complete(prompt,**kw):
        kw.update(model=model,max_tokens=min(kw.get('max_tokens',512),512),timeout=30,cache=False)
        return original_complete(prompt,**kw)
    provider.complete=bounded_complete
    for mod in (llm,embeddings,session_memory,haiku_client): mod.get_provider=lambda:provider
    # Retrieval fixture from the wiki tests: avoids embeddings/model downloads, retaining disk load and access gating.
    store=SkillStore(db_path=root/'smoke.db'); vault=root/'wiki'; vault.mkdir()
    rel='pages/entity/orchid.md'; (vault/rel).parent.mkdir(parents=True)
    (vault/rel).write_text('---\nid: orchid\nslug: orchid\ntitle: Orchid\ntype: entity\nprojects:\n  - _global\nscope: public\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\nOrchid uses port 4321. Its retry limit is 3.\n')
    store.search_vectors=lambda *args,**kw:[{'metadata':{'slug':'orchid','rel_path':rel},'score':0.9}]
    scratch_repo=root/'synthetic_repo'; scratch_repo.mkdir()
    transcript='user: Build the public Orchid demo with port 4321.\nassistant: Implemented port 4321; retry limit is 3.\nuser: Keep retry limit 3. Pending: add tests.'
    headings=['User Intent','Completed Work','Errors & Corrections','Active Work','Pending Tasks','Key References']
    checks=[]
    def run(name,fn,grade):
        current[0]=name; before=len(calls); t=time.monotonic()
        try:
            result=fn(); detail=grade(result); ok=all(detail.values())
            entries=calls[before:]; strict=bool(entries) and all(not e.get('error_type') and e.get('finish_reason') not in ('length',None) for e in entries)
            checks.append({'path':name,'duration_seconds':round(time.monotonic()-t,3),'content_checks':detail,'passed':ok and strict,'strict_transport_passed':strict})
        except Exception as exc: checks.append({'path':name,'passed':False,'error_type':type(exc).__name__})
    def facts(text): return {'port_4321':'4321' in text,'retry_3':bool(re.search(r'(retry|retries|limit).{0,35}\b3\b|\b3\b.{0,35}(retry|retries)',text,re.I))}
    run('search_summarization',lambda:searxng._summarize_results('What port and retry limit does Orchid use?',[{'title':'Orchid public demo','url':'https://example.org/orchid','snippet':'Orchid uses port 4321. Its retry limit is 3.'}]),facts)
    run('wiki_file_answer',lambda:wiki.file_answer(store,vault,'What port and retry limit does Orchid use?',top_k=1,authorized_scopes=[],model=model),lambda r:{**facts(r['answer']),'citation':'[[orchid]]' in r['answer'],'public_sources':r['sources']==['orchid'],'no_fallback':'LLM unavailable' not in r['answer']})
    run('fanout_prompt',lambda:draft_prompt(Issue(id='synthetic:1',title='Orchid validation',body='Implement port 4321 and retry limit 3. Add tests for both.',source='text'),scratch_repo,use_cache=False),lambda r:{**facts(r[0]),'tests_required':'test' in r[0].lower(),'llm_quality':r[1]=='llm'})
    run('session_memory_create',lambda:session_memory.build_session_memory(transcript,tier='mid',max_tokens=512),lambda r:{**facts(r),'six_sections':all('## '+h in r for h in headings),'pending_tests':'test' in r.lower()})
    def updated_tests_completed(text):
        # Strict six-heading structure is graded separately. Check semantic
        # stale-work claims even when a response uses numbered or HTML headings.
        normalized=re.sub(r'<[^>]+>', ' ', text)
        sections={}; active=None
        for line in normalized.splitlines():
            heading=re.sub(r'^\s*(?:#+\s*|\d+[.)]\s*)?', '', line).strip().strip('*').strip()
            match=next((h for h in headings if re.fullmatch(re.escape(h)+r'\s*:?',heading,re.I)),None)
            if match: active=match; sections.setdefault(active,[])
            elif active: sections[active].append(line)
        completed='\n'.join(sections.get('Completed Work',[]))
        pending='\n'.join(sections.get('Active Work',[])+sections.get('Pending Tasks',[]))
        affirmed=bool(re.search(r'test.{0,35}(pass|complete|done)|(?:pass|complete|done).{0,35}test',completed,re.I))
        # Any remaining test task contradicts the synthetic update unless the
        # same line explicitly says it has finished or no longer remains.
        stale=False
        for line in pending.splitlines():
            if re.search(r'\btests?\b',line,re.I) and not re.search(r'pass|complete|done|no (?:pending|remaining)|not pending',line,re.I):
                stale=True
        return affirmed and not stale
    run('session_memory_update',lambda:session_memory.update_session_memory('## User Intent\nOrchid port 4321, retry limit 3.\n## Pending Tasks\nAdd tests.','assistant: Tests now pass.\nuser: Next add the README.',tier='mid',max_tokens=512),lambda r:{**facts(r),'six_sections':all('## '+h in r for h in headings),'readme_retained':'readme' in r.lower(),'tests_completed':updated_tests_completed(r)})
    def classifier_grade(r):
        raw=re.sub(r'<think>.*?</think>','',calls[-1].get('raw',''),flags=re.S); m=re.search(r'\{.*\}',raw,re.S)
        try: data=json.loads(m.group()); c=data['classification']; schema=all(k in c for k in ('complexity','ambiguity','scope','domain_hints','confidence')) and all(isinstance(c[k],(int,float)) and 0<=c[k]<=1 for k in ('complexity','ambiguity','confidence'))
        except Exception: schema=False
        return {'raw_schema_valid':schema,'result_present':r is not None,'single_scope':r is not None and r.scope=='single','testing_domain':r is not None and 'testing' in r.domain_hints}
    run('legacy_classifier',lambda:haiku_client.classify('Write a unit test for one Python function in one repository.',conf),classifier_grade)
    def vault_snapshot():
        return {str(p.relative_to(vault)):hashlib.sha256(p.read_bytes()).hexdigest() for p in vault.rglob('*') if p.is_file()}
    before_ingest=vault_snapshot()
    def ingest_grade(r):
        raw=re.sub(r'<think>.*?</think>','',calls[-1].get('raw',''),flags=re.S); m=re.search(r'\{.*\}',raw,re.S)
        try:
            d=json.loads(m.group()); sp=d['source_page']; text=str(sp.get('body','')); schema=isinstance(sp,dict) and all(isinstance(d.get(k),list) for k in ('page_updates','index_entries','assumptions'))
        except Exception: text=''; schema=False
        return {**facts(text),'raw_schema_valid':schema,'dry_run':r.get('status')=='dry_run','public_only':all(x['scope']=='public' for x in r.get('pages',[])),'no_vault_write':vault_snapshot()==before_ingest}
    run('public_wiki_ingest',lambda:wiki.ingest_source(store,vault,source_kind='discussion',source_id='synthetic-public-orchid',source_title='Orchid public demo',source_text='Orchid uses port 4321. Its retry limit is 3.',target_scope='public',target_project='_global',authorized_scopes=[],dry_run=True,model=model),ingest_grade)
    store.close()
    report={'status':'mechanical_and_fact_checks_passed' if len(checks)==7 and all(x['passed'] for x in checks) else 'fail','scratch_root':str(root),'model':model,'calls':calls,'checks':checks,'limits':['Synthetic public facts only; raw public completions retained for human review','Substring/schema checks are bounded checks, not a full answer-quality benchmark','Transport route intercept guards model calls; socket connections are not globally monkeypatched','7 integrated calls; hard transport cap 12','512 output tokens and 30 seconds each','Output truncation is failure','Wiki retrieval vectors are fixture hits; embedding quality is untested','No live registry or session-memory persistence','Wiki ingest dry-run; canonical write durability is untested']}
    output=a.output.resolve() if a.output else root/'report.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); sys.exit(0 if report['status']=='mechanical_and_fact_checks_passed' else 1)
