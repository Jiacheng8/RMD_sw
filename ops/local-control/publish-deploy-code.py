"""Local code-only backup/deploy; no weights or credentials enter the archive."""
import base64,datetime,hashlib,json,os,pathlib,subprocess,sys
BASE=pathlib.Path(os.environ.get('RDM_MAC_WORKSPACE', '/Users/shaoshitong/Documents/Codex/2026-11-02/chat-thread-codex-threads-01a19ad1-3f77-2')).resolve()
REPO=BASE/'work/h20-main-publication-source'
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)
sha=git('rev-parse','HEAD').decode().strip()
remote=git('ls-remote','git@github.com:shaoshitong/rdm_lmmd.git','refs/heads/main').decode().split()[0]
assert remote==sha
subprocess.run(['git','fetch','--no-tags','git@github.com:shaoshitong/rdm_lmmd.git','refs/heads/main'],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
assert git('rev-parse','FETCH_HEAD').decode().strip()==sha
backup=BASE/'backups'/('h20-code-docs-2026-11-04-'+sha[:7]);backup.mkdir(exist_ok=True)
files=[]
for item in git('ls-tree','-rz',sha).split(b'\0'):
 if not item:continue
 info,name=item.split(b'\t',1);mode,kind,oid=info.decode().split();name=name.decode()
 assert kind=='blob';data=git('cat-file','blob',oid)
 p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(int(mode,8)&0o777)
 digest=hashlib.sha256(data).hexdigest();assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 files.append(dict(path=name,mode=mode,bytes=len(data),sha256=digest))
manifest=dict(source_commit=sha,repository='shaoshitong/rdm_lmmd',branch='main',files=files)
(backup/'CODE-DOCS-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
receipt=dict(repository='shaoshitong/rdm_lmmd',branch='main',local_commit=sha,remote_commit=remote,files_verified=len(files),code_docs_backup=str(backup),verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_weights_or_environment=True)
(BASE/'outputs/h20-local-code-docs-backup.json').write_text(json.dumps(receipt,indent=2)+'\n')
if len(sys.argv)>1:
 dest=sys.argv[1];assert dest.startswith(('/data/rdm-workspace/RDM-', '/data/shaoshitong/RDM-'))
 payload=base64.b64encode(git('archive','--format=tar.gz',sha)).decode()
 script='''import base64,hashlib,io,json,pathlib,tarfile,os
D=pathlib.Path(DEST);assert not D.exists();D.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as t:
 for m in t.getmembers():
  assert not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
 t.extractall(D,filter='data')
for f in FILES:
 p=D/f['path'];assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
 assert (p.stat().st_mode&0o777)==(int(f['mode'],8)&0o777)
(D/'SOURCE_COMMIT').write_text(SHA+'\\n')
print(json.dumps(dict(source=str(D),source_commit=SHA,files_verified=len(FILES))))
'''
 script= 'DEST='+repr(dest)+'\nPAYLOAD='+repr(payload)+'\nFILES='+repr(files)+'\nSHA='+repr(sha)+'\n'+script
 r=subprocess.run([sys.executable,str(BASE/'work/h20-recovery-ssh.py')],input=script,text=True,capture_output=True,check=True)
 deployment=json.loads(r.stdout);receipt['deployment']=deployment
(BASE/'outputs/h20-latest-code-publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
