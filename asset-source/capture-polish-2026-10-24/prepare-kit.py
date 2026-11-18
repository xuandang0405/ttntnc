"""Read-only geometry analysis. Image packing is handled by pack-kit.ps1."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
NAMES = {'tốt':'pawn','mã':'knight','tượng':'bishop','xe':'rook','hậu':'queen','vua':'king'}
HEIGHTS = {'tốt':210,'mã':250,'tượng':250,'xe':225,'hậu':260,'vua':270}
manifest = json.loads((ROOT/'img/animation-fix-manifest.json').read_text(encoding='utf-8-sig'))

def geometry(image, frame):
    a = np.array(image.getchannel('A').crop((frame['x'],0,frame['x']+frame['width'],frame['height'])))
    labels, n = ndimage.label(a >= 230)
    if not n:
        raise ValueError('No solid piece found')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    ys, xs = np.nonzero(labels == sizes.argmax())
    body = [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    edge_y = body[3] - max(4,round((body[3]-body[1])*.08))
    base_x = float(np.median(xs[ys>=edge_y]))
    yy, xx = np.nonzero(a >= 64)
    content = [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    return {'rootX':base_x,'rootY':body[3],'bodyHeight':body[3]-body[1],'bodyBounds':body,'contentBounds':content}

jobs=[]
for entry in manifest['animations']:
    if entry['action'] not in ('move','attack','die'):
        continue
    image=Image.open(ROOT/entry['file']).convert('RGBA')
    if entry['action']=='die':
        geometries=[geometry(image,entry['frames'][0])]*4
    else:
        geometries=[geometry(image,f) for f in entry['frames']]
    scale=HEIGHTS[entry['piece']]/geometries[0]['bodyHeight']
    side='white' if entry['color']=='trắng' else 'black'
    frames=[]
    for i,(frame,geo) in enumerate(zip(entry['frames'],geometries)):
        bob=[0,-6,-12,-6][i] if entry['action']=='move' else 0
        lunge=[0,3,5,0][i] if entry['action']=='attack' else 0
        frames.append({**frame,'drawX':160+lunge-geo['rootX']*scale,'drawY':316+bob-geo['rootY']*scale,'scale':scale,'geometry':geo})
    jobs.append({'source':entry['file'],'target':f'img/animations/{side}/{entry["action"]}_{NAMES[entry["piece"]]}.png','color':side,'piece':NAMES[entry['piece']],'action':entry['action'],'frames':frames,'frameDurationsMs':entry['frameDurationsMs'],'loop':entry['action']=='move','hideAfterPlayback':entry['action']=='die'})

expected={(side,piece,action) for side in ('white','black') for piece in NAMES.values() for action in ('move','attack','die')}
assert {(j['color'],j['piece'],j['action']) for j in jobs}==expected, 'Missing piece states'
(SOURCE/'kit-layout.json').write_text(json.dumps({'frameWidth':320,'frameHeight':360,'anchor':{'x':160,'y':316},'jobs':jobs},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Prepared {len(jobs)} normalized sheets')
