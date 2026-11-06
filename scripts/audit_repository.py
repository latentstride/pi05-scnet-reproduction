#!/usr/bin/env python3
"""Read-only repository inventory, syntax and local Markdown link audit.

Uses only the standard library. Does not import project modules, install
packages, load models, run shell/Slurm entry points, or contact the network.
Writes JSON only when --output is explicitly supplied.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


def tracked(root):
    return subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=root
    ).decode().rstrip('\0').split('\0')


def markdown_links(root, names):
    missing=[]
    checked=0
    for name in names:
        path=root/name
        if path.suffix!='.md' or not path.is_file():continue
        # Fenced code is an example, not a rendered Markdown link.
        text=re.sub(r'```[\s\S]*?```', lambda m:'\n'*m[0].count('\n'), path.read_text())
        for match in re.finditer(r'(?<!!)\[[^\]\n]+\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)', text):
            target=unquote(match[1].strip('<>')).split('#',1)[0].split('?',1)[0]
            if not target or re.match(r'[a-z][a-z0-9+.-]*:',target,re.I):continue
            checked+=1
            resolved=(root/target.lstrip('/')) if target.startswith('/') else path.parent/target
            if not resolved.exists():
                missing.append(dict(source=name,line=text[:match.start()].count('\n')+1,target=target))
    return dict(checked=checked,missing=missing,
        scope='rendered local file links only; no remote URLs, anchors or runtime assets')


def scan(root):
    names=[n for n in tracked(root) if n and (root/n).is_file()]
    texts={};hashes=defaultdict(list);imports={};syntax=[];sensitive=[];private_paths=[]
    script_names=[n for n in names if (n.startswith('scripts/') or n.startswith('research/') or '/' not in n)
                  and Path(n).suffix in ('.py','.sh','.slurm')]
    for name in names:
        p=root/name;data=p.read_bytes();hashes[hashlib.sha256(data).hexdigest()].append(name)
        if b'\0' in data[:8192]:continue
        try:text=data.decode('utf-8')
        except UnicodeDecodeError:continue
        texts[name]=text
        for line_no,line in enumerate(text.splitlines(),1):
            if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',line):
                sensitive.append(dict(path=name,line=line_no,kind='credential-pattern; value suppressed'))
            if re.search(r'/(?:Users|public/home|home)/[A-Za-z0-9_.-]+/',line):
                private_paths.append(dict(path=name,line=line_no,kind='deployment/user path; value suppressed'))
        if p.suffix=='.py':
            try:tree=ast.parse(text,filename=name)
            except SyntaxError as error:
                syntax.append(dict(path=name,line=error.lineno,error=error.msg));continue
            modules=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):modules.extend(a.name.split('.')[0] for a in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module:modules.append(node.module.split('.')[0])
            imports[name]=sorted(set(modules))
    inventory=[]
    for name in script_names:
        p=Path(name);module=p.stem;text=texts.get(name,'')
        callers=sorted(n for n,mods in imports.items() if n!=name and module in mods)
        references=sorted(n for n,t in texts.items() if n!=name and p.name in t and n not in callers)
        tests=sorted(n for n in set(callers+references) if n.startswith('tests/'))
        doc=''
        if p.suffix=='.py':
            try:doc=(ast.get_docstring(ast.parse(text)) or '').split('\n')[0]
            except SyntaxError:pass
        if not doc:
            doc=next((line.lstrip('#! ').strip() for line in text.splitlines()
                      if line.startswith('#') and not line.startswith('#!')),'unknown')
        if name.startswith('research/'):
            status='historical';stage='archived snapshot'
        elif module.startswith('g2_remaining'):
            status='experimental';stage='G2 remaining CPU research'
        elif module.startswith('g2e') or 'g2e' in module:
            status='experimental';stage='G2-E'
        elif re.search(r'_v[2-9](?:$|_)',module):
            status='historical';stage='G2 archived diagnostic version'
        elif module.startswith('g2'):
            status='active' if callers or tests else 'unknown';stage='G2'
        elif module.startswith(('pace_g1','prepare_pace','pace_client','pace_runtime')):
            status='historical';stage='G1 interface acceptance'
        elif any(k in module for k in ('libero','evaluate','closed_loop','inference')):
            status='historical';stage='Stage 1 frozen LIBERO'
        elif module.startswith(('fetch','install','download','pull','run_container','validate','verify_checkpoint')):
            status='unknown';stage='environment/download'
        elif 'robotwin' in module or 'archive' in module:
            status='experimental';stage='Stage 2 RoboTwin data/environment'
        else:status='unknown';stage='unknown'
        inventory.append(dict(path=name,purpose=doc,stage=stage,status=status,
            usage_evidence='static references present' if callers or references else 'unknown',
            callers=callers,other_references=references,test_associations=tests,
            coverage='static test association only' if tests else 'unknown; no direct static association found',
            resource_entry=bool(p.suffix=='.slurm' or re.search(r'^\s*(?:exec\s+)?(?:sbatch|srun|salloc)\b',text,re.M))))
    duplicates=[dict(sha256=h,paths=paths) for h,paths in hashes.items() if len(paths)>1]
    versions=defaultdict(list)
    for name in script_names:
        base=re.sub(r'_v\d+(?=\.|_)','_vN',Path(name).name)
        if base!=Path(name).name:versions[base].append(name)
    large=sorted((dict(path=n,bytes=(root/n).stat().st_size) for n in names if (root/n).stat().st_size>=1024**2),key=lambda r:-r['bytes'])
    return dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        tracked_file_count=len(names),tracked_bytes=sum((root/n).stat().st_size for n in names),
        python_files_checked=len(imports)+len(syntax),python_syntax_errors=syntax,
        script_inventory=inventory,duplicate_content_groups=duplicates,version_families=dict(versions),
        imports_by_file=imports,
        large_files_over_1mib=large,credential_pattern_findings=sensitive,
        deployment_path_findings=private_paths,markdown=markdown_links(root,names),
        limitations=['static associations are not executed coverage',
            'dynamic imports/runtime-generated paths need deployment checks',
            'regex scan is not a comprehensive secrets audit; archive payloads not scanned',
            'no GPU/model/Slurm or external environment checks were executed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-docs',nargs='*',metavar='PATH')
    args=parser.parse_args();root=args.root.resolve()
    if args.check_docs is not None:
        result=markdown_links(root,args.check_docs or tracked(root))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        raise SystemExit(1 if result['missing'] else 0)
    result=scan(root)
    if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('base_commit','tracked_file_count','python_files_checked',
        'python_syntax_errors','large_files_over_1mib','credential_pattern_findings')},ensure_ascii=False,indent=2))
