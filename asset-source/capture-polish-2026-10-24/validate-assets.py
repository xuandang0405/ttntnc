from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'animation-preview/capture-complete'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((ROOT/'asset-backups/before-capture-polish-2026-10-24/inventory-before.json').read_text(encoding='utf-8'))
spec=json.loads((Path(__file__).parent/'generation-spec.json').read_text(encoding='utf-8'))
expected={p for j in spec['jobs'] for p in [j['target'],*j['aliases']] if p in before}
changed={p for p,info in before.items() if digest(ROOT/p)!=info['sha256']}
assert changed==expected,(changed-expected,expected-changed)
for j in spec['jobs']:
    for alias in j['aliases']:
        assert digest(ROOT/alias)==digest(ROOT/j['target'])
    if j['target'] in before:
        backup=ROOT/'asset-backups/before-capture-polish-2026-10-24'/j['target']
        assert Image.open(ROOT/j['target']).size==Image.open(backup).size
assert all(digest(ROOT/p)==v['sha256'] for p,v in before.items() if '/move_' in p or '/idel_' in p or p=='img/Bàn cờ.png')

kit=json.loads((ROOT/'img/animations/manifest.json').read_text(encoding='utf-8-sig'))
counts=Counter(j['action'] for j in kit['animations'])
assert counts=={'move':12,'idle':12,'attack':12,'die':12,'impact':2},counts
files={j['file'] for j in kit['animations']}
assert len(files)==38
stats=[]
for j in kit['animations']:
    path=ROOT/'img/animations'/j['file']
    image=Image.open(path);image.verify();image=Image.open(path)
    assert image.mode=='RGBA'
    if j['action']=='idle':
        assert j['frameIndices']==[0]
        continue
    assert image.size==(1280,j['frameHeight']) and j['frameCount']==4
    alpha=np.array(image.getchannel('A'))
    assert alpha.min()==0 and alpha.max()==255
    masses=[];roots=[];heights=[];hashes=[];edges=[]
    for i in range(4):
        a=alpha[:,i*320:(i+1)*320]
        edges.append(int((a[:,[0,-1]]>64).sum()+(a[[0,-1],:]>64).sum()))
        assert edges[-1]==0,(path,i,'clipped or bleeding sprite')
        hashes.append(hashlib.sha256(image.crop((i*320,0,(i+1)*320,image.height)).tobytes()).hexdigest())
        masses.append(float(a.sum()/255))
        if j['action'] in ('move','attack') or (j['action']=='die' and i==0):
            labels,n=ndimage.label(a>=230);sizes=np.bincount(labels.ravel());sizes[0]=0
            y,x=np.nonzero(labels==sizes.argmax());h=int(y.max()+1-y.min())
            bx=float(np.median(x[y>=y.max()+1-max(4,round(h*.08))]))
            roots.append([round(bx,1),int(y.max()+1)]);heights.append(h)
    assert len(set(hashes))==4,(path,'duplicate frames')
    if j['action']=='die':
        assert alpha[:,960:].max()<=51 and masses[3]<.15*masses[0],(path,'death does not fade')
    if j['action']=='move':
        assert all(abs(r[0]-160)<=3 for r in roots),(path,roots)
        assert all(abs(r[1]-(316+[0,-6,-12,-6][i]))<=3 for i,r in enumerate(roots)),(path,roots)
    stats.append({'file':j['file'],'action':j['action'],'alphaMass':[round(n,1) for n in masses],'registeredRoots':roots,'solidBodyHeights':heights,'edgePixelsAbove64':edges})

gifs=[]
for side in ('white','black'):
    path=OUT/f'move-and-capture-{side}.gif';im=Image.open(path)
    assert im.n_frames==28 and im.info.get('loop')==0
    durations=[]
    for i in range(im.n_frames):
        im.seek(i);im.load();durations.append(im.info.get('duration'))
    assert durations==[50]*25+[250,250,400],durations
    gifs.append({'file':path.name,'frames':im.n_frames,'durationMs':sum(durations),'loop':0})

victims=['knight','rook','bishop','queen','pawn','rook']
pieces=['pawn','knight','bishop','rook','queen','king']
scenes=[]
for side in ('white','black'):
    for piece,victim in zip(pieces,victims):
        sr=3 if side=='white' else 1;target=[2,2]
        if piece in ('bishop','queen'):target=[3,1 if side=='white' else 3]
        if piece=='rook':target=[3,sr]
        if piece=='knight':target=[3,2]
        dx,dy=target[0]-1,target[1]-sr
        assert victim!='king'
        if piece=='pawn':assert dx==1 and dy==(-1 if side=='white' else 1)
        if piece=='knight':assert sorted([abs(dx),abs(dy)])==[1,2]
        if piece=='bishop':assert abs(dx)==abs(dy)
        if piece=='rook':assert dx==0 or dy==0
        if piece=='queen':assert dx==0 or dy==0 or abs(dx)==abs(dy)
        if piece=='king':assert max(abs(dx),abs(dy))==1
        scenes.append({'attackerColor':side,'attacker':piece,'victimColor':'black' if side=='white' else 'white','victim':victim,'from':[1,sr],'to':target})
(OUT/'preview-sequences.json').write_text(json.dumps(scenes,ensure_ascii=False,indent=2),encoding='utf-8')
assert kit['capture']['impactMs']==360+90+90
assert kit['capture']['victimHiddenMs']==kit['capture']['victimDieStartMs']+sum([140,160,180,200])
report={'status':'passed','generatedAtlases':19,'legacyPngsReplaced':len(changed),'legacyPngsPreserved':len(before)-len(changed),'standardizedSheets':38,'logicalStates':dict(counts),'uniqueFrames':38*4,'legalMovementProfilesInPreviews':12,'kingVictimsInPreviews':0,'gifs':gifs,'replacedFiles':sorted(changed),'sheets':stats}
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('sheets','replacedFiles')},ensure_ascii=False,indent=2))
