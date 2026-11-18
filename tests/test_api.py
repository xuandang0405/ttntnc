import io
import chess.pgn
import pytest
from fastapi.testclient import TestClient
from backend.api.config import Settings
from backend.api.server import create_app

@pytest.fixture
def client(tmp_path):
    app=create_app(Settings(database=str(tmp_path/'games.sqlite3')))
    with TestClient(app) as c:yield c

def new(client,**options):
    r=client.post('/api/reset',json=options);assert r.status_code==200,r.text
    return r.json()

def action(client,game,path,**extra):
    return client.post('/api/'+path,headers={'Authorization':'Bearer '+game['token']},json={'game_id':game['game_id'],'revision':game['state']['revision'],**extra})

def advance(client,game,path,**extra):
    r=action(client,game,path,**extra);assert r.status_code==200,r.text
    return {**r.json(),'token':game['token']}

def test_health_isolation_validation_and_stale_revision(client):
    assert client.get('/health').json()['levels']==[1,2,3,4]
    assert client.get('/ready').status_code==200
    a=new(client);b=new(client)
    assert client.get('/api/games/'+a['game_id']).status_code==401
    assert client.get('/api/games/'+a['game_id'],headers={'Authorization':'Bearer '+b['token']}).status_code==401
    assert action(client,a,'human-move',uci='e2e5').status_code==422
    assert action(client,a,'ai-move',level=5).status_code==422
    moved=advance(client,a,'human-move',uci='e2e4')
    assert moved['state']['ply']==1 and moved['move']['from']=='e2'
    assert action(client,a,'human-move',uci='d2d4').status_code==409
    assert action(client,moved,'human-move',uci='d2d4').json()['error']['code']=='WRONG_TURN'
    assert client.post('/api/reset',json={'fen':'not a fen'}).status_code==422

def test_random_replay_and_round_undo(client):
    a=new(client,seed=77);b=new(client,seed=77)
    for key in ['a','b']:
        g=a if key=='a' else b
        g=advance(client,g,'human-move',uci='e2e4');g=advance(client,g,'ai-move',level=1,time_ms=100)
        if key=='a':a=g
        else:b=g
    assert a['state']['fen']==b['state']['fen']
    choice=a['move']['uci']
    a=advance(client,a,'undo');assert a['state']['ply']==0
    a=advance(client,a,'human-move',uci='e2e4');a=advance(client,a,'ai-move',level=1)
    assert a['move']['uci']==choice and a['telemetry']['nodes']==0

def test_black_player_pause_finish_and_undo(client):
    g=new(client,human_color='black',seed=3)
    assert action(client,g,'human-move',uci='e7e5').status_code==409
    g=advance(client,g,'ai-move',level=1);assert not g['state']['can_undo']
    g=advance(client,g,'pause',paused=True)
    assert action(client,g,'human-move',uci=g['state']['legal_actions'][0]).json()['error']['code']=='GAME_PAUSED'
    g=advance(client,g,'pause',paused=False)
    g=advance(client,g,'human-move',uci=g['state']['legal_actions'][0]);g=advance(client,g,'ai-move',level=1)
    g=advance(client,g,'finish',kind='resign');assert g['state']['terminal'] and g['state']['result']==1
    g=advance(client,g,'undo');assert g['state']['ply']==1 and not g['state']['terminal']

@pytest.mark.parametrize('fen,uci,capture_square,rook,promotion',[
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1','e5d6','d5',None,None),
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','e1g1',None,{'from':'h1','to':'f1'},None),
    ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1','a7a8r',None,None,'rook'),
])
def test_animation_events_for_special_moves(client,fen,uci,capture_square,rook,promotion):
    g=new(client,mode='local',fen=fen)
    g=advance(client,g,'human-move',uci=uci)
    assert (g['move']['captured']['square'] if g['move']['captured'] else None)==capture_square
    assert g['move']['rook']==rook and g['move']['promotion']==promotion
    g=advance(client,g,'undo');assert g['state']['fen']==fen

def test_pgn_hint_and_persistence(tmp_path):
    settings=Settings(database=str(tmp_path/'persist.sqlite3'))
    with TestClient(create_app(settings)) as client:
        g=new(client,mode='local',seed=41)
        for uci in ['e2e4','d7d5','e4d5']:g=advance(client,g,'human-move',uci=uci)
        headers={'Authorization':'Bearer '+g['token']}
        hint=client.get('/api/games/'+g['game_id']+'/hint',headers=headers).json()
        assert hint['uci'] in g['state']['legal_actions']
        pgn=client.get('/api/games/'+g['game_id']+'/pgn',headers=headers).text
        restored=chess.pgn.read_game(io.StringIO(pgn)).end().board()
        assert restored.fen()==g['state']['fen']
        assert g['state']['captured']['black']==['pawn']
    with TestClient(create_app(settings)) as client:
        state=client.get('/api/games/'+g['game_id'],headers=headers).json()['state']
        assert state['fen']==g['state']['fen'] and state['revision']==g['state']['revision']

@pytest.mark.parametrize('level',[1,2,3,4])
def test_difficulty_saved_and_uses_real_opponent(client,level):
    g=new(client,level=level,human_color='black',seed=19)
    assert g['state']['level']==level
    g=advance(client,g,'ai-move',level=level)
    assert g['state']['ply']==1 and g['state']['turn']=='black'
    assert g['telemetry']['algorithm']==('Random' if level==1 else 'Greedy' if level==2 else 'AlphaBeta')
    assert g['telemetry']['nodes']==0 if level==1 else g['telemetry']['nodes']>0
    assert client.post('/api/reset',json={'level':5}).status_code==422

def test_level_cannot_change_mid_game(client):
    g=new(client,level=3,human_color='black')
    assert action(client,g,'ai-move',level=1).json()['error']['code']=='LEVEL_MISMATCH'
