import type {Color,Kind} from './types'
export type Action='idle'|'move'|'attack'|'die'|'impact'
export type Entry={file:string;action:Action;piece:Kind|'effect';color:Color;frameWidth:number;frameHeight:number;frameCount:number;frameIndices?:number[];loop:boolean;anchor:{x:number;y:number};frameDurationsMs:number[]}
export type Bundle={entries:Map<string,Entry>;images:Map<string,HTMLImageElement>}
let cached:Promise<Bundle>|null=null
export function loadAssets(){
 if(!cached)cached=(async()=>{
  const response=await fetch('/animations/manifest.json',{cache:'no-store'});if(!response.ok)throw new Error('Không thể tải bộ quân cờ.')
  const manifest=await response.json() as {animations:Entry[]}
  const entries=new Map(manifest.animations.map(e=>[`${e.color}/${e.piece}/${e.action}`,e]))
  const images=new Map<string,HTMLImageElement>()
  await Promise.all([...new Set(manifest.animations.map(e=>e.file))].map(file=>new Promise<void>((resolve,reject)=>{const image=new Image();image.onload=()=>{images.set(file,image);resolve()};image.onerror=()=>reject(new Error('Không thể tải ảnh quân cờ.'));image.src=`/animations/${file}`})))
  return {entries,images}
 })().catch(e=>{cached=null;throw e})
 return cached
}
export function spriteUrl(color:Color,kind:Kind,action:Action='move'){return `/animations/${color}/${action}_${kind}.png`}
