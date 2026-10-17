import {ArrowRight,BookOpen,CheckCircle,Diamond,DoorOpen,LockKey,Play,ShieldWarning} from '@phosphor-icons/react';
import {difficultyName} from './types';
import type {Difficulty,Game,Level,Metrics} from './types';

export function LevelSelect({levels,metrics,currentId,busy,onSelect}:{levels:Level[];metrics:Metrics;currentId?:number;busy:boolean;onSelect:(id:number)=>void}){
  return <div className="level-list">{levels.map(level=>{
    const unlocked=metrics.unlocked_levels.includes(level.id),completed=metrics.completed_levels.includes(level.id);
    return <button key={level.id} className={`level-card ${currentId===level.id?'selected':''} ${unlocked?'':'locked'}`} disabled={busy||!unlocked}
      title={unlocked?`Chơi màn ${level.id}`:`Chiến thắng màn ${level.id-1} để mở khóa`}
      aria-label={`Màn ${level.id}: ${level.name}${unlocked?'':', đang khóa'}`} aria-pressed={currentId===level.id} onClick={()=>onSelect(level.id)}>
      <div className="level-image" style={{backgroundImage:`url(${level.image})`}}><span>{String(level.id).padStart(2,'0')}</span>
        {!unlocked?<span className="locked-label"><LockKey size={16} weight="fill"/>ĐANG KHÓA</span>:completed?<CheckCircle className="completed-mark" size={23} weight="fill"/>:<span className="current-map">{currentId===level.id?'ĐANG CHƠI':'SẴN SÀNG'}</span>}
      </div><div className="level-caption"><h3>{level.name}</h3><p>{unlocked?completed?'Đã hoàn thành · Có thể chơi lại':level.subtitle:`Thắng màn ${level.id-1} để mở khóa`}</p></div>
    </button>;
  })}</div>;
}

export function Menu({levels,metrics,savedGame,busy,aiLevel,onDifficulty,onStart,onContinue,onGuide}:{levels:Level[];metrics:Metrics;savedGame:Game|null;busy:boolean;aiLevel:Difficulty;onDifficulty:(level:Difficulty)=>void;onStart:(id:number)=>void;onContinue:()=>void;onGuide:()=>void}){
  const canContinue=savedGame?.state.status==='playing'&&savedGame.state.rules_version===metrics.rules_version&&metrics.unlocked_levels.includes(savedGame.state.level_id);
  return <div className="main-menu">
    <section className="menu-hero" style={{backgroundImage:"url('/assets/Backgroud.png')"}} aria-label="Menu bắt đầu trò chơi">
      <div className="menu-shade"/>
      <div className="menu-content"><p className="eyebrow">MỘT SIÊU TRỘM · NĂM PHI VỤ · MỘT LỐI THOÁT</p>
        <h1 className="sr-only">Shadow Heist — Phi vụ bóng đêm</h1><img className="menu-logo" src="/assets/NameGame.png" alt="Shadow Heist"/>
        <p className="menu-tagline">Lấy kim cương. Tránh tia laser. Biến mất trong bóng tối.</p>
        <fieldset className="difficulty-picker" disabled={busy}><legend>ĐỘ KHÓ WARDEN</legend><div>{([1,2,3] as const).map(value=><button key={value} type="button" className={value===3?'nightmare-choice':undefined} aria-pressed={aiLevel===value} onClick={()=>onDifficulty(value)}><strong>{difficultyName(value)}</strong><span>{value===1?'Ngẫu nhiên':value===2?'Đã huấn luyện':'Thử thách tối đa'}</span></button>)}</div><p>{aiLevel===3?'Tính trước mọi phản công trong 60 nửa lượt.':aiLevel===2?'Đuổi theo bạn thông minh hơn một chút.':'Làm quen với các phi vụ đầu tiên.'}</p></fieldset>
        <div className="menu-buttons"><button className="primary-button menu-start" disabled={busy} onClick={()=>onStart(metrics.campaign_complete?1:metrics.next_level)}><Play size={20} weight="fill"/>{busy?'Đang chuẩn bị…':'Bắt đầu trò chơi'}<ArrowRight size={19}/></button>
          {canContinue&&<button className="menu-continue" disabled={busy} onClick={onContinue}>Tiếp tục màn {savedGame.state.level_id}<span>{difficultyName(savedGame.ai_level)} · {savedGame.state.turn}/60 nửa lượt</span></button>}
          <button className="menu-help" onClick={onGuide}><BookOpen size={17}/> Hướng dẫn chơi</button>
        </div>
        <div className="menu-rules"><span><Diamond size={17}/>Thu thập đủ 2 kim cương</span><span><ShieldWarning size={17}/>Chạm laser là thua</span><span><DoorOpen size={17}/>Về EXIT để chiến thắng</span></div>
      </div>
      <div className="menu-bottomline"><span>{aiLevel===3?'ÁC MỘNG / KHÔNG KHOAN NHƯỢNG':aiLevel===2?'DỄ+ / WARDEN ĐÃ HUẤN LUYỆN':'DỄ / ROBOT NGẪU NHIÊN'}</span><span>WASD · PHÍM MŨI TÊN · CHẠM Ô</span></div>
    </section>
    <section className="menu-campaign"><div className="campaign-heading"><div><div className="eyebrow">HÀNH TRÌNH CỦA SHADOW</div><h2>Mở khóa từng phi vụ.</h2><p>Chiến thắng màn trước để bước vào màn tiếp theo.</p></div><div className="campaign-count"><span>{String(metrics.completed_levels.length).padStart(2,'0')}<small> / 05</small></span><p>NHIỆM VỤ HOÀN THÀNH</p></div></div>
      <LevelSelect levels={levels} metrics={metrics} busy={busy} onSelect={onStart}/>
      {metrics.campaign_complete&&<p className="campaign-finished"><CheckCircle size={18} weight="fill"/>Bạn đã hoàn thành cả chiến dịch. Tất cả màn đều có thể chơi lại.</p>}
    </section>
  </div>;
}
