"""Update playback metadata only; original raster sprites are left unchanged."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for relative in ('img/animations/manifest.json','frontend/public/animations/manifest.json'):
    path=ROOT/relative
    manifest=json.loads(path.read_text(encoding='utf-8-sig'))
    manifest['schemaVersion']=3
    for entry in manifest['animations']:
        if entry['action']=='idle':
            entry.update(frameCount=4,frameIndices=[0,1,2,3],frameDurationsMs=[240,280,240,280],loop=True)
        elif entry['action']=='attack':
            entry['frameDurationsMs']=[140,140,140,180]
        elif entry['action']=='die':
            entry['frameDurationsMs']=[170,200,230,260]
        elif entry['action']=='impact':
            entry['frameDurationsMs']=[70,90,120,160]
    manifest['motion'].update(durationMs=700,minDurationMs=650,maxDurationMs=1100)
    manifest['capture'].update(travelMs=480,attackStartMs=480,impactMs=760,victimDieStartMs=760,attackerArrivesMs=1160,victimHiddenMs=1620,completeMs=1760)
    manifest['notes']=[note for note in manifest['notes'] if 'Idle is frame zero' not in note and 'not the chess rules' not in note]
    note='Idle loops all four registered move frames with its own timing; the game adds breathing, base aura and slow motes without changing piece identity.'
    if note not in manifest['notes']:
        manifest['notes'].append(note)
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(relative)
