import {memo,useEffect,useRef,useState} from 'react';
import type {Cell,Game,Level} from './types';

const SIZE=1254, PAD=80, TILE=156.3;
const center=([r,c]:Cell):Cell=>[PAD+(c+.5)*TILE,PAD+(r+.5)*TILE];
const same=(a:Cell,b:Cell)=>a[0]===b[0]&&a[1]===b[1];
type Motion={from:Cell;to:Cell;path:Cell[];start:number;duration:number};

function drawActor(ctx:CanvasRenderingContext2D,img:HTMLImageElement,pos:Cell,running:boolean,frame:number,flip:boolean,robot=false,scale=1){
  // Source rectangles exclude the borders printed around the supplied sprite sheet.
  const sx=(robot?29:24)+frame*355, sy=running?(robot?568:585):(robot?96:125);
  const sw=326,sh=running?(robot?410:391):(robot?440:420);
  const h=TILE*1.03*scale,w=h*sw/sh;
  ctx.save();ctx.translate(pos[0],pos[1]);
  ctx.fillStyle='rgba(4,9,18,.4)';ctx.beginPath();ctx.ellipse(0,TILE*.35*scale,w*.38,11*scale,0,0,Math.PI*2);ctx.fill();
  if(flip)ctx.scale(-1,1);
  ctx.imageSmoothingEnabled=false;
  ctx.drawImage(img,sx,sy,sw,sh,-w/2,-h*.62,w,h);
  ctx.restore();
}

export const Operative=memo(function Operative({paused=false}:{paused?:boolean}){
  const ref=useRef<HTMLCanvasElement>(null);
  useEffect(()=>{
    const img=new Image();img.src='/assets/shadow.png';let raf=0,disposed=false;
    const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
    function draw(now:number){
      const ctx=ref.current?.getContext('2d');
      if(ctx&&img.complete&&img.naturalWidth){ctx.clearRect(0,0,320,320);drawActor(ctx,img,[160,169],false,paused||reduced?0:Math.floor(now/230)%4,false,false,1.6);}
      if(!disposed)raf=requestAnimationFrame(draw);
    }
    raf=requestAnimationFrame(draw);
    return()=>{disposed=true;cancelAnimationFrame(raf);};
  },[paused]);
  return <canvas ref={ref} width="320" height="320" className="operative-art" aria-label="Shadow, siêu trộm với animation đứng"/>;
});

export const Board=memo(function Board({level,game,paused,disabled,heatmap,telemetry,onMove}:{level:Level;game:Game;paused:boolean;disabled:boolean;heatmap:boolean;telemetry:Game['telemetry'];onMove:(cell:Cell)=>void}){
  const ref=useRef<HTMLCanvasElement>(null);
  const snapshot=useRef({level,game,paused,disabled,heatmap,telemetry});
  const motion=useRef<{thief:Motion;warden:Motion}|null>(null);
  const [error,setError]=useState(false);
  useEffect(()=>{
    const old=snapshot.current.game;
    const now=performance.now();
    if(!motion.current||old.game_id!==game.game_id){
      const make=(p:Cell):Motion=>({from:p,to:p,path:[p],start:now,duration:0});
      motion.current={thief:make(game.state.thief_pos),warden:make(game.state.warden_pos)};
    }else{
      for(const [key,pos] of [['thief',game.state.thief_pos],['warden',game.state.warden_pos]] as const){
        const previous=motion.current[key].to;
        if(!same(previous,pos)){
          const path=game.path.length>1?game.path:[previous,pos];
          motion.current[key]={from:previous,to:pos,path,start:now,duration:Math.max(340,(path.length-1)*260)};
        }
      }
    }
    snapshot.current={level,game,paused,disabled,heatmap,telemetry};
  },[level,game,paused,disabled,heatmap,telemetry]);

  useEffect(()=>{
    let disposed=false,raf=0,lastTime=0;
    const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
    const images:Record<string,HTMLImageElement>={};
    for(const [key,src] of Object.entries({thief:'/assets/shadow.png',warden:'/assets/robot.png',...Object.fromEntries([1,2,3,4,5].map(i=>[`map${i}`,`/assets/level_${i}.png`]))})){
      const img=new Image();img.onerror=()=>{if(!disposed)setError(true);};img.src=src;images[key]=img;
    }
    function frame(now:number){
      const ctx=ref.current?.getContext('2d');
      const {level,game,paused,disabled,heatmap,telemetry}=snapshot.current;
      const state=game.state, map=images[`map${level.id}`];
      if(paused&&motion.current&&lastTime){for(const m of Object.values(motion.current))m.start+=now-lastTime;}
      lastTime=now;
      const time=paused||reduced?0:now;
      if(ctx&&map.complete&&map.naturalWidth){
        ctx.clearRect(0,0,SIZE,SIZE);ctx.drawImage(map,0,0,SIZE,SIZE);
        for(const [r,c] of level.floor_patches){
          // The original artwork seals routes. A visible maintenance tile opens them.
          ctx.drawImage(map,PAD+4*TILE,PAD+6*TILE,TILE,TILE,PAD+c*TILE,PAD+r*TILE,TILE,TILE);
          ctx.fillStyle='rgba(126,219,199,.08)';ctx.fillRect(PAD+c*TILE,PAD+r*TILE,TILE,TILE);
          ctx.strokeStyle='#83c9ba';ctx.lineWidth=3;ctx.setLineDash([9,12]);ctx.strokeRect(PAD+c*TILE+15,PAD+r*TILE+15,TILE-30,TILE-30);ctx.setLineDash([]);
        }
        // Exit is visible at all times; it opens only after both diamonds are collected.
        const [ex,ey]=center(level.exit),unlocked=state.collected.every(Boolean);
        ctx.save();ctx.translate(ex,ey);ctx.strokeStyle=unlocked?'#a7f3d0':'#9eb6d3';ctx.fillStyle=unlocked?'rgba(100,230,180,.2)':'rgba(10,20,35,.56)';ctx.lineWidth=4;
        ctx.fillRect(-45,-50,90,100);ctx.strokeRect(-45,-50,90,100);ctx.fillStyle=ctx.strokeStyle;ctx.font='600 20px sans-serif';ctx.textAlign='center';ctx.fillText('EXIT',0,7);ctx.restore();
        for(let r=0;r<7;r++)for(let c=0;c<7;c++)if(level.grid[r][c]===2){
          const [x,y]=center([r,c]);ctx.save();ctx.translate(x,y);
          ctx.fillStyle='rgba(232,103,103,.12)';ctx.fillRect(-TILE*.43,-TILE*.43,TILE*.86,TILE*.86);
          ctx.strokeStyle=`rgba(255,115,127,${.65+Math.sin(time/700)*.2})`;ctx.lineWidth=4;
          for(const offset of [-28,0,28]){ctx.beginPath();ctx.moveTo(-TILE*.38,offset);ctx.lineTo(TILE*.38,offset);ctx.stroke();}
          ctx.fillStyle='#ffadac';for(const side of [-1,1])ctx.fillRect(side*TILE*.38-4,-39,8,78);ctx.restore();
        }
        if(heatmap&&telemetry){
          for(const [key,value] of Object.entries(telemetry.action_scores)){
            const [r,c]=key.split(',').map(Number),[x,y]=center([r,c]);
            ctx.fillStyle='rgba(230,190,123,.25)';ctx.fillRect(x-TILE*.44,y-TILE*.44,TILE*.88,TILE*.88);
          }
        }
        if(state.player===1&&state.status==='playing'&&!paused&&!disabled){
          for(const action of state.legal_actions){
            const [x,y]=center(action);ctx.strokeStyle='rgba(159,235,218,.8)';ctx.fillStyle='rgba(159,235,218,.1)';ctx.lineWidth=3;
            ctx.fillRect(x-60,y-60,120,120);ctx.strokeRect(x-60,y-60,120,120);
            ctx.fillStyle='#c5f4e8';ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fill();
          }
        }
        level.diamonds.forEach((cell,i)=>{
          if(state.collected[i])return;
          const [x,y]=center(cell),dy=Math.sin(time/650+i)*5;
          ctx.save();ctx.translate(x,y+dy);ctx.fillStyle='rgba(154,224,238,.15)';ctx.beginPath();ctx.ellipse(0,39,31,9,0,0,Math.PI*2);ctx.fill();
          ctx.beginPath();ctx.moveTo(0,-39);ctx.lineTo(29,-10);ctx.lineTo(0,34);ctx.lineTo(-29,-10);ctx.closePath();ctx.fillStyle='#b8f4f0';ctx.fill();ctx.strokeStyle='#eefefb';ctx.lineWidth=3;ctx.stroke();
          ctx.beginPath();ctx.moveTo(0,-39);ctx.lineTo(8,-10);ctx.lineTo(0,34);ctx.lineTo(-8,-10);ctx.closePath();ctx.fillStyle='#57abae';ctx.fill();ctx.restore();
        });
        for(const key of ['thief','warden'] as const){
          const m=motion.current?.[key],img=images[key];if(!m||!img.complete||!img.naturalWidth)continue;
          const progress=reduced?1:Math.min(1,Math.max(0,(now-m.start)/m.duration||1));
          const segment=Math.min(m.path.length-2,Math.floor(progress*(m.path.length-1)));
          let pos=center(m.to);
          if(progress<1&&segment>=0){
            const a=center(m.path[segment]),b=center(m.path[segment+1]);
            const t=progress*(m.path.length-1)-segment;
            pos=[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];
          }
          const frameIndex=paused||reduced?0:Math.floor(now/(progress<1?90:230))%4;
          drawActor(ctx,img,pos,progress<1,frameIndex,m.to[1]<m.from[1],key==='warden');
        }
        // Keep probability labels readable when the selected square contains an actor.
        if(heatmap&&telemetry){
          for(const [key,value] of Object.entries(telemetry.action_scores)){
            const [r,c]=key.split(',').map(Number),[x,y]=center([r,c]);
            ctx.fillStyle='rgba(12,18,29,.9)';ctx.fillRect(x-42,y+33,84,30);
            ctx.fillStyle='#ffe2ac';ctx.font='bold 24px sans-serif';ctx.textAlign='center';ctx.fillText(`${Math.round(value*100)}%`,x,y+55);
          }
        }
      }
      if(!disposed)raf=requestAnimationFrame(frame);
    }
    raf=requestAnimationFrame(frame);
    return()=>{disposed=true;cancelAnimationFrame(raf);};
  },[]);

  return <div className="board-surface">
    <canvas ref={ref} width={SIZE} height={SIZE} className="game-canvas" aria-hidden="true"/>
    <div className="cell-inputs" role="group" aria-label="Bản đồ 7 hàng 7 cột. Chọn ô được đánh dấu để đi.">
      {Array.from({length:49},(_,i)=>{const r=Math.floor(i/7),c=i%7,legal=game.state.player===1&&game.state.legal_actions.some(a=>same(a,[r,c]));return <button key={i} type="button" tabIndex={legal?0:-1} disabled={disabled||!legal||paused||game.state.status!=='playing'} aria-label={`Ô ${'ABCDEFG'[c]}${r+1}${level.grid[r][c]===1?', tường':level.grid[r][c]===2?', laser - chạm vào là thua':same([r,c],level.exit)?`, cổng EXIT ${game.state.collected.every(Boolean)?'đã mở':'cần đủ 2 kim cương'}`:''}${legal?', có thể di chuyển':''}`} onClick={()=>onMove([r,c])}/>;})}
    </div>
    {error&&<div className="asset-error">Không tải được ảnh game. Hãy tải lại trang.</div>}
  </div>;
});
