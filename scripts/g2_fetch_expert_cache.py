#!/usr/bin/env python3
"""Fetch only small cached planner members from the fixed official expert ZIPs."""
import argparse,hashlib,json,ssl,zipfile
from pathlib import Path
import sample_robotwin_archives as fetch

def run(manifest,output,endpoint):
    data=json.loads(manifest.read_text());output.mkdir(parents=True,exist_ok=False)
    fetch.context=lambda:ssl.create_default_context(cafile='/etc/ssl/cert.pem')
    records=[]
    for row in data['archives']:
        if row['robot']!='aloha-agilex':continue
        url='https://'+endpoint+'/datasets/'+data['repository']+'/resolve/'+data['revision']+'/'+row['archive_path']
        remote=fetch.RangeFile(url,row['archive_bytes'],4*1024*1024)
        with zipfile.ZipFile(remote) as z:
            for i in range(3):
                name='aloha-agilex_clean_50/_traj_data/episode'+str(i)+'.pkl'
                info=z.getinfo(name)
                if info.file_size>1024*1024:raise ValueError('unexpected planner member size')
                payload=z.read(info);file=output/Path(name).name;file.write_bytes(payload)
                records.append(dict(file=file.name,archive_member=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),zip_crc32='%08x'%info.CRC,crc_verified=True))
        result=dict(endpoint=endpoint,repository=data['repository'],revision=data['revision'],archive_path=row['archive_path'],full_archive_sha256_verified=False,http_bytes_received=remote.transferred,range_requests=remote.requests,members=records)
        (output/'provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--endpoint',choices=['huggingface.co','hf-mirror.com'],default='huggingface.co');a=p.parse_args();run(a.manifest,a.output,a.endpoint)
