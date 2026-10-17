import React,{useCallback,useEffect,useRef,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {ArrowClockwise,ArrowDown,ArrowLeft,ArrowRight,ArrowUp,ArrowUpRight,BookOpen,CheckCircle,Clock,Code,Diamond,DoorOpen,Flag,GridFour,House,ListBullets,Pause,Play,Robot,ShieldWarning,Target,X} from '@phosphor-icons/react';
import '@fontsource/be-vietnam-pro/latin-400.css';
import '@fontsource/be-vietnam-pro/vietnamese-400.css';
import '@fontsource/be-vietnam-pro/latin-600.css';
import '@fontsource/be-vietnam-pro/vietnamese-600.css';
import '@fontsource/be-vietnam-pro/latin-700.css';
import '@fontsource/be-vietnam-pro/vietnamese-700.css';
import {api,ApiError} from './api';
import {Board,Operative} from './Board';
import {Menu,LevelSelect} from './Menu';
import {cellLabel,difficultyName} from './types';
import type {Cell,Difficulty,Game,History,Level,Metrics,Telemetry} from './types';
import './style.css';
import './menu.css';
import './difficulty.css';

const delay=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
const STATUS:Record<string,string>={playing:'Đang chơi',won:'Chiến thắng!',lost:'Nhiệm vụ thất bại',draw:'Hòa',abandoned:'Đã đổi nhiệm vụ',outdated:'Ván theo luật cũ'};
function App(){
  const [screen,setScreen]=useState<'menu'|'playing'>('menu');
  const [aiLevel,setAiLevel]=useState<Difficulty>(()=>{const saved=Number(localStorage.getItem('shadow_ai_level'));return saved===1||saved===2?saved:3;});
  const chooseDifficulty=(value:Difficulty)=>{setAiLevel(value);localStorage.setItem('shadow_ai_level',String(value));};
  const [levels,setLevels]=useState<Level[]>([]),[game,setGame]=useState<Game|null>(null);
  const [busy,setBusy]=useState(false),[paused,setPaused]=useState(false),[heatmap,setHeatmap]=useState(false);
  const [error,setError]=useState(''),[loading,setLoading]=useState(true),[guide,setGuide]=useState(false),[historyOpen,setHistoryOpen]=useState(false);
  const [history,setHistory]=useState<History[]>([]),[historyLoading,setHistoryLoading]=useState(false);
  const [metrics,setMetrics]=useState<Metrics>({games:0,wins:0,draws:0,completed_levels:[],unlocked_levels:[1],next_level:1,campaign_complete:false,rules_version:2});
  const [telemetry,setTelemetry]=useState<Telemetry|null>(null);
  const [events,setEvents]=useState<string[]>(['Kết nối với phòng điều khiển.']);
  const lock=useRef(false),mounted=useRef(true);
  const level=levels.find(l=>l.id===game?.state.level_id);
  const remember=(g:Game)=>{setGame(g);localStorage.setItem('shadow_game',g.game_id);};
  const log=(message:string)=>setEvents(e=>[message,...e].slice(0,5));
  const refreshMetrics=useCallback(async()=>{
    const progress=await api<Metrics>('/metrics');setMetrics(progress);
    setLevels(maps=>maps.map(l=>({...l,unlocked:progress.unlocked_levels.includes(l.id),completed:progress.completed_levels.includes(l.id)})));
    return progress;
  },[]);

  const initialize=useCallback(async()=>{
    setLoading(true);setError('');
    try{
      await api('/player',{});
      const maps=await api<Level[]>('/levels');if(!mounted.current)return;setLevels(maps);
      const progress=await refreshMetrics();
      const saved=localStorage.getItem('shadow_game');let g:Game|null=null;
      if(saved){try{g=await api<Game>(`/games/${saved}`);}catch(e){if(!(e instanceof ApiError)||e.status!==404)throw e;}}
      if(g&&(g.state.rules_version!==progress.rules_version||g.state.status==='abandoned'||!progress.unlocked_levels.includes(g.state.level_id)))g=null;
      if(mounted.current){setGame(g);setTelemetry(null);setScreen('menu');log('Shadow đã sẵn sàng. Bắt đầu từ menu trò chơi.');}
    }catch(e){if(mounted.current)setError((e as Error).message);}
    finally{if(mounted.current)setLoading(false);}
  },[refreshMetrics]);
  useEffect(()=>{mounted.current=true;void initialize();return()=>{mounted.current=false;};},[initialize]);
  useEffect(()=>{window.scrollTo({top:0,left:0,behavior:'instant'});},[screen]);
  useEffect(()=>{
    if(!guide&&!historyOpen)return;
    const previous=document.activeElement as HTMLElement|null;
    const modal=document.querySelector<HTMLElement>('.modal');
    const controls=()=>Array.from(modal?.querySelectorAll<HTMLElement>('button:not(:disabled),a[href]')??[]);
    controls()[0]?.focus();
    const trap=(event:KeyboardEvent)=>{
      if(event.key!=='Tab')return;
      const items=controls(),first=items[0],last=items[items.length-1];
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
    };
    document.addEventListener('keydown',trap);
    return()=>{document.removeEventListener('keydown',trap);previous?.focus();};
  },[guide,historyOpen]);

  const recover=useCallback(async(e:unknown)=>{
    setError((e as Error).message);
    if(game&&e instanceof ApiError&&e.status===409){
      try{remember(await api<Game>(`/games/${game.game_id}`));}catch{/* Original error remains visible. */}
    }
  },[game]);

  const humanMove=useCallback(async(cell:Cell)=>{
    if(screen!=='playing'||!game||lock.current||paused||guide||historyOpen||game.state.player!==1||game.state.status!=='playing')return;
    if(!game.state.legal_actions.some(a=>a[0]===cell[0]&&a[1]===cell[1]))return;
    lock.current=true;setBusy(true);setError('');
    try{
      const next=await api<Game>('/human-move',{game_id:game.game_id,expected_turn:game.state.turn,row:cell[0],col:cell[1]});remember(next);
      const found=next.state.collected.filter(Boolean).length-game.state.collected.filter(Boolean).length;
      log(next.state.reason==='laser'?'Đã chạm laser. Phi vụ thất bại.':found?next.state.collected.every(Boolean)?'Đã đủ 2 kim cương! Hãy trở về EXIT tại A1.':'Đã lấy được một viên kim cương.':next.state.status==='won'?'Đã thoát qua EXIT. Chiến thắng!':`Shadow di chuyển tới ${cellLabel(cell)}.`);
      await delay(480);
      if(next.state.status!=='playing')await refreshMetrics();
    }catch(e){await recover(e);}finally{lock.current=false;setBusy(false);}
  },[game,paused,guide,historyOpen,recover,screen,refreshMetrics]);

  useEffect(()=>{
    if(screen!=='playing'||!game||game.state.player!==-1||game.state.status!=='playing'||paused||busy||guide||historyOpen||error)return;
    const id=setTimeout(async()=>{
      if(lock.current)return;lock.current=true;setBusy(true);
      try{
        const next=await api<Game>('/ai-move',{game_id:game.game_id,expected_turn:game.state.turn,time_ms:100});
        remember(next);setTelemetry(next.telemetry);log(`Warden tuần tra tới ${cellLabel(next.state.warden_pos)}.`);
        await delay(600);if(next.state.status!=='playing')await refreshMetrics();
      }catch(e){await recover(e);}finally{lock.current=false;setBusy(false);}
    },450);
    return()=>clearTimeout(id);
  },[game,paused,busy,guide,historyOpen,error,recover,screen,refreshMetrics]);

  const reset=async(id:number)=>{
    if(!metrics.unlocked_levels.includes(id)){setError(`Hãy thắng màn ${id-1} để mở khóa màn ${id}.`);return;}
    if(lock.current)return;lock.current=true;setBusy(true);setError('');
    try{
      const next=await api<Game>('/reset',{level_id:id,ai_level:screen==='playing'&&game?game.ai_level:aiLevel,previous_game_id:game?.game_id});remember(next);setTelemetry(null);setPaused(false);setScreen('playing');
      setEvents([`Nhiệm vụ ${String(id).padStart(2,'0')} bắt đầu. Lấy 2 kim cương rồi trở về EXIT.`]);await refreshMetrics();
    }catch(e){await recover(e);}finally{lock.current=false;setBusy(false);}
  };
  const returnToMenu=()=>{if(lock.current)return;setScreen('menu');setPaused(false);setError('');};
  const togglePause=useCallback(()=>setPaused(p=>!p),[]);
  useEffect(()=>{
    const handler=(event:KeyboardEvent)=>{
      if(event.target instanceof HTMLInputElement||event.target instanceof HTMLTextAreaElement)return;
      if(event.key==='Escape'){setGuide(false);setHistoryOpen(false);return;}
      if(guide||historyOpen||screen!=='playing')return;
      const directions:Record<string,Cell>={ArrowUp:[-1,0],w:[-1,0],ArrowRight:[0,1],d:[0,1],ArrowDown:[1,0],s:[1,0],ArrowLeft:[0,-1],a:[0,-1]};
      const delta=directions[event.key.length===1?event.key.toLowerCase():event.key];
      if(delta&&game){event.preventDefault();void humanMove([game.state.thief_pos[0]+delta[0],game.state.thief_pos[1]+delta[1]]);}
      if(event.code==='Space'&&event.target===document.body){event.preventDefault();togglePause();}
    };
    window.addEventListener('keydown',handler);return()=>window.removeEventListener('keydown',handler);
  },[game,humanMove,guide,historyOpen,togglePause,screen]);
  const openHistory=async()=>{
    setHistoryOpen(true);setHistoryLoading(true);setError('');
    try{setHistory(await api<History[]>('/games'));}catch(e){setError((e as Error).message);}finally{setHistoryLoading(false);}
  };
  const resume=async(id:string)=>{
    if(lock.current)return;lock.current=true;setBusy(true);
    try{const g=await api<Game>(`/games/${id}`);if(g.state.rules_version!==metrics.rules_version)throw new Error('Ván này thuộc luật cũ. Hãy bắt đầu ván mới.');remember(g);chooseDifficulty(g.ai_level);setTelemetry(null);setHistoryOpen(false);setPaused(false);setScreen('playing');setError('');log('Đã tải lại ván chơi từ MySQL.');}catch(e){setError((e as Error).message);}finally{lock.current=false;setBusy(false);}
  };
  const state=game?.state;
  return <div className="app-shell">
    <header className="topbar" inert={guide||historyOpen}>
      <button className="brand" disabled={busy} onClick={returnToMenu} aria-label="Shadow Heist — về menu chính"><img className="header-logo" src="/assets/NameGame.png" alt="Shadow Heist"/></button>
      <nav className="desktop-nav" aria-label="Điều hướng"><button disabled={busy} onClick={returnToMenu} className={screen==='menu'?'nav-active':''}><House size={17}/>Menu chính</button><button onClick={()=>setGuide(true)}><BookOpen size={17}/>Cách chơi</button><a href="/project-docs/" target="_blank" rel="noreferrer">Hồ sơ đồ án<ArrowUpRight size={14}/></a></nav>
      <div className="topbar-right"><span className="connection"><i/> {levels.length?'Hệ thống trực tuyến':'Đang kết nối'}</span>{screen==='playing'&&<button className="icon-button mobile-home" disabled={busy} aria-label="Về menu chính" onClick={returnToMenu}><House size={20}/></button>}<button className="icon-button" aria-label="Lịch sử chơi" title="Lịch sử chơi" disabled={busy||!levels.length} onClick={()=>void openHistory()}><Clock size={20}/></button><div className="avatar">SH</div></div>
    </header>

    <main className={screen==='menu'?'menu-root':undefined} inert={guide||historyOpen}>
      {screen==='playing'&&<section className="intro"><div><div className="eyebrow"><span/> CHIẾN DỊCH / PHI VỤ BÓNG ĐÊM</div><h1>Đột nhập. Thu thập. <em>Biến mất.</em></h1><p>Một siêu trộm, một robot tuần tra. Bạn sẽ tìm được lối thoát?</p></div><div className="campaign-count"><span>{String(metrics.completed_levels.length).padStart(2,'0')}<small> / 05</small></span><p>NHIỆM VỤ HOÀN THÀNH</p></div></section>}

      {error&&<div className="error-banner" role="alert"><ShieldWarning size={20}/><span>{error}</span><button onClick={()=>game?setError(''):void initialize()}>{game?'Thử tiếp':'Kết nối lại'}</button></div>}
      {loading?<div className="loading-layout" aria-label="Đang tải game"><div/><div/><div/></div>:screen==='menu'&&levels.length?<Menu aiLevel={aiLevel} onDifficulty={chooseDifficulty} levels={levels} metrics={metrics} savedGame={game} busy={busy} onStart={id=>void reset(id)} onContinue={()=>game&&void resume(game.game_id)} onGuide={()=>setGuide(true)}/>:game&&level&&state?<>
        <section className="game-layout">
          <aside className="left-column">
            <section className="operative panel"><div className="section-label">ĐẶC VỤ CỦA BẠN <span className="tiny-dot"/></div><div className="portrait"><div className="portrait-grid"/><Operative paused={paused||guide||historyOpen}/><span className="portrait-code">PH–01</span></div><div className="operative-name"><h2>Shadow</h2><span>SIÊU TRỘM</span></div><p>Không để lại dấu vết.<br/>Chỉ mang theo kim cương.</p><div className="role-badge"><i/> BẠN ĐI TRƯỚC</div></section>
            <section className="controls panel"><div className="section-label">ĐIỀU KHIỂN</div><div className="keypad" aria-label="Phím di chuyển"><kbd>W</kbd><div><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd></div></div><p>WASD / phím mũi tên<br/>hoặc chọn ô sáng trên bản đồ</p><button className="text-button" onClick={()=>setGuide(true)}>Xem luật chơi <ArrowUpRight size={14}/></button></section>
            <div className="left-note"><ShieldWarning size={17}/><p>Chạm vào laser là thua.<br/>Tìm đường an toàn tới EXIT.</p></div>
          </aside>

          <section className="arena panel" aria-label="Khu vực chơi">
            <div className="arena-top"><div><span className="mission-number">NHIỆM VỤ {String(level.id).padStart(2,'0')}</span><h2>{level.name}</h2></div><span className={`difficulty ${state.alert?'alert-badge':game.ai_level===3?'nightmare-badge':''}`}>{state.alert?<ShieldWarning size={15}/>:<span className="tiny-dot"/>}{state.alert?'BÁO ĐỘNG':`CHẾ ĐỘ ${difficultyName(game.ai_level).toLocaleUpperCase('vi-VN')}`}</span></div>
            {state.collected.every(Boolean)&&state.status==='playing'&&<div className="exit-open-banner" role="status"><DoorOpen size={17}/>Đã đủ kim cương! Đi vào EXIT tại A1 để chiến thắng.</div>}
            <div className="board-zone"><div className="board-columns">{'ABCDEFG'.split('').map(c=><span key={c}>{c}</span>)}</div><div className="board-rows">{[1,2,3,4,5,6,7].map(r=><span key={r}>{r}</span>)}</div><Board level={level} game={game} disabled={busy} paused={paused||guide||historyOpen} heatmap={heatmap} telemetry={telemetry} onMove={cell=>void humanMove(cell)}/>
              {paused&&state.status==='playing'&&<div className="board-overlay"><Pause size={38}/><h2>Phi vụ tạm dừng</h2><p>Hít thở. Quan sát. Tính bước tiếp theo.</p><button className="primary-button" onClick={togglePause}><Play size={16} weight="fill"/>Tiếp tục phi vụ</button></div>}
              {state.status!=='playing'&&!busy&&<div className={`board-overlay result-overlay ${state.status==='won'?'victory':''}`} role="status"><div className="result-icon">{state.status==='won'?<Diamond size={42} weight="duotone"/>:state.status==='draw'?<Flag size={42}/>:<ShieldWarning size={42}/>}</div><div className="eyebrow">PHI VỤ {String(level.id).padStart(2,'0')}</div><h2>{STATUS[state.status]}</h2><p>{state.status==='won'?'Đã thu thập đủ 2 kim cương và thoát qua EXIT.':state.reason==='laser'?'Shadow đã chạm vào tia laser. Hãy tìm lối đi an toàn hơn.':state.reason==='captured'?'Warden đã bắt được Shadow. Hãy thử một lộ trình khác.':state.status==='draw'?'Hết 60 nửa lượt. Bạn đã giữ được ít nhất một kim cương.':state.status==='outdated'?'Ván này thuộc luật cũ. Hãy bắt đầu ván mới từ menu.':'Hết thời gian. Hãy thử một lộ trình khác.'}</p>{state.status==='won'&&<div className="win-unlocked"><CheckCircle size={17} weight="fill"/>{level.id<5?`Đã mở khóa màn ${level.id+1}`:'Đã hoàn thành cả 5 màn!'}</div>}<button className="primary-button" onClick={()=>state.status==='outdated'?returnToMenu():void reset(state.status==='won'&&level.id<5?level.id+1:level.id)}>{state.status==='outdated'?'Về menu chính':state.status==='won'&&level.id<5?`Chơi màn ${level.id+1}`:'Chơi lại màn này'}<ArrowRight size={17}/></button><button className="result-menu-button" onClick={returnToMenu}><House size={15}/>Menu chính</button></div>}
            </div>
            <div className="arena-bottom"><div className={`turn-indicator ${state.player===-1?'robot-turn':''}`}><i/>{state.status!=='playing'?STATUS[state.status]:paused?'Đã tạm dừng':state.player===1?'Lượt của bạn':'Warden đang tuần tra…'}</div><div className="arena-actions"><button className="icon-button" disabled={state.status!=='playing'} onClick={togglePause} aria-label={paused?'Tiếp tục':'Tạm dừng'} title={paused?'Tiếp tục':'Tạm dừng'}>{paused?<Play size={18}/>:<Pause size={18}/>}</button><button className="icon-button" disabled={busy} onClick={()=>void reset(level.id)} aria-label="Chơi lại màn này" title="Chơi lại màn này"><ArrowClockwise size={18}/></button></div></div>
            <div className="mobile-pad">{[[ArrowLeft,[0,-1]],[ArrowUp,[-1,0]],[ArrowDown,[1,0]],[ArrowRight,[0,1]]].map(([Icon,delta],i)=>{const I=Icon as typeof ArrowLeft;const d=delta as Cell;return <button key={i} aria-label={['Sang trái','Đi lên','Đi xuống','Sang phải'][i]} disabled={busy||paused||state.player!==1} onClick={()=>void humanMove([state.thief_pos[0]+d[0],state.thief_pos[1]+d[1]])}><I size={24}/></button>;})}</div>
          </section>

          <aside className="right-column">
            <section className="objectives panel"><div className="section-label"><Target size={16}/> MỤC TIÊU PHI VỤ</div><div className="diamond-counter"><Diamond size={25} weight="duotone"/><strong>{state.collected.filter(Boolean).length}<span> / 2</span></strong><span>kim cương</span></div><div className="objective-list">{level.diamonds.map((cell,i)=><div key={i} className={state.collected[i]?'completed':''}>{state.collected[i]?<CheckCircle size={19} weight="fill"/>:<span className="objective-circle"/>}<span>Thu thập kim cương {i+1}</span><small>{cellLabel(cell)}</small></div>)}<div className={state.status==='won'?'completed':''}>{state.status==='won'?<CheckCircle size={19} weight="fill"/>:<DoorOpen size={19}/>}<span>Trở về cổng thoát</span><small>A1</small></div></div><div className="turn-budget"><div><span>GIỚI HẠN NỬA LƯỢT</span><strong>{String(state.turn).padStart(2,'0')}<small> / 60</small></strong></div><div className="budget-track"><span style={{transform:`scaleX(${state.turn/60})`}}/></div></div></section>
            <section className="robot-panel panel"><div className="section-label"><Robot size={17}/> ĐỐI THỦ <span className="live-label">LIVE</span></div><div className="robot-heading"><div className="robot-avatar"><Robot size={29} weight="duotone"/></div><div><h3>Warden Bot</h3><p>{game.ai_level===3?'Model + Minimax · Ác mộng':game.ai_level===2?'Đã huấn luyện · Dễ+':'Random Agent · Dễ'}</p></div></div><div className="algorithm-note">{game.ai_level===3?'Truy đuổi, chặn mục tiêu và tận dụng thời gian còn lại. Mỗi lượt chỉ đi một ô.':game.ai_level===2?'Đã học cách đuổi theo Shadow; vẫn có nước đi ngẫu nhiên. Mỗi lượt chỉ đi một ô.':'Chọn ngẫu nhiên đều từ các ô hợp lệ. Mỗi nước đi là một khả năng.'}</div><div className="telemetry"><div><span>Ô có thể chọn</span><strong>{telemetry?.nodes??'—'}</strong></div><div><span>Thời gian quyết định</span><strong>{telemetry?`${telemetry.latency_ms.toFixed(2)} ms`:'—'}</strong></div></div><button className={`heatmap-toggle ${heatmap?'is-on':''}`} role="switch" aria-checked={heatmap} onClick={()=>setHeatmap(h=>!h)}><span><GridFour size={16}/> Xem xác suất nước đi</span><i><span/></i></button>{heatmap&&<p className="heatmap-note">{game.ai_level===3?'Ác mộng chọn nước tối ưu một cách cố định: ô được chọn có xác suất 100%.':game.ai_level===2?'Xác suất của lượt Warden gần nhất, kết hợp dự đoán model và ngẫu nhiên.':'Xác suất trước nước đi gần nhất. Các ô có xác suất bằng nhau, không phải điểm chiến thuật.'}</p>}</section>
            <section className="activity"><div className="section-label"><ListBullets size={16}/> NHẬT KÝ PHI VỤ</div><ol aria-live="polite">{events.slice(0,3).map((event,i)=><li key={`${i}-${event}`}><i/><span>{event}</span></li>)}</ol></section>
          </aside>
        </section>

        <section className="campaign"><div className="campaign-heading"><div className="section-label">CHỌN NHIỆM VỤ <span>05 BẢN ĐỒ / 01 CHIẾN DỊCH</span></div><p>Chiến thắng để mở khóa màn tiếp theo.</p></div><LevelSelect levels={levels} metrics={metrics} currentId={level.id} busy={busy} onSelect={id=>{if(id!==level.id)void reset(id);}}/></section>
        <footer><span><span className="tiny-dot"/> SHADOW HEIST <b>/</b> Đồ án Trí tuệ nhân tạo nâng cao</span><span>MAP {level.id} <b>·</b> SEED {game.seed} <b>·</b> v1.3</span></footer>
      </>:<section className="offline-state"><Robot size={42}/><h2>Phòng điều khiển chưa sẵn sàng</h2><p>Khởi động dịch vụ game rồi kết nối lại.</p><button className="primary-button" onClick={()=>void initialize()}>Kết nối lại <ArrowClockwise size={18}/></button></section>}
    </main>
    {(guide||historyOpen)&&<div className="modal-backdrop" onClick={()=>{setGuide(false);setHistoryOpen(false);}}><section className="modal panel" role="dialog" aria-modal="true" aria-labelledby="modal-title" onClick={e=>e.stopPropagation()}><button autoFocus className="icon-button close-modal" aria-label="Đóng" onClick={()=>{setGuide(false);setHistoryOpen(false);}}><X size={22}/></button>{guide?<><div className="eyebrow">HỒ SƠ ĐẶC VỤ / 001</div><h2 id="modal-title">Nghệ thuật biến mất.</h2><p className="modal-intro">Thu thập đủ 2 viên kim cương và trở về cổng EXIT tại A1 trước khi Warden bắt được bạn.</p><div className="guide-rule"><Diamond size={26}/><div><h3>01 / Lấy kim cương</h3><p>Đi tới ô kim cương để thu thập. Chưa đủ cả hai thì cổng chưa mở.</p></div></div><div className="guide-rule"><GridFour size={26}/><div><h3>02 / Di chuyển theo lượt</h3><p>Shadow đi trước, mỗi lượt một ô theo 4 hướng. Robot đi sau. Dùng WASD, mũi tên hoặc chạm vào ô sáng.</p></div></div><div className="guide-rule"><ShieldWarning size={26}/><div><h3>03 / Cẩn thận với laser</h3><p>Chạm vào laser là thua ngay lập tức. Kim cương luôn nằm trên ô sàn an toàn; robot tuần tra một bước mỗi lượt.</p></div></div><div className="guide-rule"><Clock size={26}/><div><h3>04 / 60 nửa lượt</h3><p>Một nước đi của mỗi nhân vật tính là một nửa lượt. Hết 60: có ít nhất một kim cương thì hòa; chưa có kim cương thì thua. Trùng ô với Warden là bị bắt.</p></div></div><a className="docs-link" href="/project-docs/" target="_blank" rel="noreferrer"><Code size={18}/>Sơ đồ hệ thống, thuật toán và schema SQL<ArrowUpRight size={16}/></a></>:<><div className="eyebrow">DỮ LIỆU ĐƯỢC LƯU TRONG MYSQL</div><h2 id="modal-title">Những phi vụ đã qua.</h2>{historyLoading?<p>Đang tải lịch sử…</p>:history.length===0?<p>Chưa có ván chơi nào. Bắt đầu phi vụ đầu tiên của bạn.</p>:<div className="history-list">{history.map(h=><button key={h.id} disabled={h.status==='abandoned'||h.rules_version!==metrics.rules_version||busy} onClick={()=>void resume(h.id)}><div><strong>{levels.find(l=>l.id===h.level_id)?.name??`Màn ${h.level_id}`}</strong><small>{new Date(h.created_at).toLocaleString('vi-VN')} · {difficultyName(h.ai_level)} · {h.turn} nửa lượt · {h.collected}/2 kim cương</small></div><span className={h.status==='won'?'text-accent':''}>{h.rules_version!==metrics.rules_version?'Ván theo luật cũ':STATUS[h.status]}<ArrowUpRight size={14}/></span></button>)}</div>}</>}</section></div>}
  </div>;
}
createRoot(document.getElementById('root')!).render(<App/>);
