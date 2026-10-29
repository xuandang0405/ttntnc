import {useEffect,useState} from 'react'
import {motionEnabled} from './motion'
import type {AnimationMode} from './motion'

export function useMotion(mode:AnimationMode){
 const [reduced,setReduced]=useState(()=>window.matchMedia('(prefers-reduced-motion: reduce)').matches)
 useEffect(()=>{
  const media=window.matchMedia('(prefers-reduced-motion: reduce)')
  const change=()=>setReduced(media.matches)
  media.addEventListener('change',change)
  return()=>media.removeEventListener('change',change)
 },[])
 return {animated:motionEnabled(mode,reduced),systemReduced:reduced}
}
