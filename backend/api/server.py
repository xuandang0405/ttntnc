from contextlib import asynccontextmanager
import json
import logging
import time
from uuid import uuid4
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from backend.api.config import ROOT, Settings
from backend.api.errors import AppError
from backend.api.games.repository import GameRepository
from backend.api.games.schemas import NewGame, HumanMove, AiMove, GameAction, PauseAction, FinishAction
from backend.api.games.service import GameService

log = logging.getLogger('ky-dai')
logging.basicConfig(level=logging.INFO,format='%(message)s')

def create_app(settings: Settings | None = None):
    settings = settings or Settings()
    repo = GameRepository(settings.database_path)
    service = GameService(repo,settings.max_ply)

    @asynccontextmanager
    async def lifespan(app):
        repo.migrate()
        yield

    app = FastAPI(title='Kỳ Đài Chess API',version='1.0.0',lifespan=lifespan)
    app.state.service = service
    app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins,allow_methods=['GET','POST'],allow_headers=['Content-Type','Authorization','X-Request-ID'])

    @app.middleware('http')
    async def request_context(request: Request, call_next):
        request.state.request_id = uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers['X-Request-ID'] = request.state.request_id
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        if request.url.path.startswith('/api'):
            response.headers['Cache-Control'] = 'no-store'
            log.info(json.dumps({'request_id':request.state.request_id,'method':request.method,'path':request.url.path,'status':response.status_code,'latency_ms':round((time.perf_counter()-start)*1000,2)}))
        return response

    @app.exception_handler(AppError)
    async def app_error(request, exc):
        return JSONResponse(status_code=exc.status,content={'error':{'code':exc.code,'message':exc.message},'request_id':request.state.request_id})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422,content={'error':{'code':'VALIDATION_ERROR','message':'Dữ liệu không hợp lệ. Vui lòng kiểm tra lại.','fields':[{'field':'.'.join(str(p) for p in e['loc']),'message':e['msg']} for e in exc.errors()]},'request_id':request.state.request_id})

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        log.exception('Unhandled server error',exc_info=exc)
        return JSONResponse(status_code=500,content={'error':{'code':'INTERNAL_ERROR','message':'Máy chủ gặp lỗi. Bạn có thể thử lại.'},'request_id':getattr(request.state,'request_id','')})

    def token(authorization):
        return authorization[7:] if authorization and authorization.startswith('Bearer ') else ''

    @app.get('/health')
    def health():
        return {'status':'ok','app':'co-vua','algorithm':'Random / Greedy / AlphaBeta','levels':[1,2,3,4]}

    @app.get('/ready')
    def ready():
        with repo.connect() as db:
            db.execute('SELECT 1 FROM games LIMIT 1')
        return {'status':'ready'}

    @app.post('/api/reset')
    def reset(body: NewGame):
        return service.new(body)

    @app.get('/api/games/{game_id}')
    def state(game_id: str, authorization: str | None = Header(default=None)):
        return service.get(game_id,token(authorization))

    @app.post('/api/human-move')
    def human(body: HumanMove, authorization: str | None = Header(default=None)):
        return service.play(body.game_id,token(authorization),body.revision,uci=body.uci)

    @app.post('/api/ai-move')
    def ai(body: AiMove, authorization: str | None = Header(default=None)):
        return service.play(body.game_id,token(authorization),body.revision,ai=True,time_ms=body.time_ms,level=body.level)

    @app.post('/api/undo')
    def undo(body: GameAction, authorization: str | None = Header(default=None)):
        return service.undo(body.game_id,token(authorization),body.revision)

    @app.post('/api/pause')
    def pause(body: PauseAction, authorization: str | None = Header(default=None)):
        return service.pause(body.game_id,token(authorization),body.revision,body.paused)

    @app.post('/api/finish')
    def finish(body: FinishAction, authorization: str | None = Header(default=None)):
        return service.finish(body.game_id,token(authorization),body.revision,body.kind)

    @app.get('/api/games/{game_id}/hint')
    def hint(game_id: str, authorization: str | None = Header(default=None)):
        return service.hint(game_id,token(authorization))

    @app.get('/api/games/{game_id}/pgn')
    def pgn(game_id: str, authorization: str | None = Header(default=None)):
        return PlainTextResponse(service.pgn(game_id,token(authorization)),media_type='application/x-chess-pgn',headers={'Content-Disposition':'attachment; filename="ky-dai.pgn"'})

    @app.get('/api/tournament-results')
    def tournaments():
        return {'rows':[],'available':False,'message':'Bản này triển khai cấp Dễ Random. Chưa chạy giải đấu.'}

    dist = ROOT/'frontend/dist'
    if (dist/'index.html').exists():
        app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')
    else:
        @app.get('/')
        def frontend_help():
            return {'message':'Dùng frontend dev tại http://127.0.0.1:5173 hoặc chạy npm run build trong frontend.'}
    return app

app = create_app()
