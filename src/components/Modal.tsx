import {useEffect,useRef} from 'react'
import type {ReactNode} from 'react'
import {X} from '@phosphor-icons/react'

export function Modal({title,eyebrow,onClose,children,wide=false}:{title:string;eyebrow?:string;onClose:()=>void;children:ReactNode;wide?:boolean}){
 const ref=useRef<HTMLDialogElement>(null)
 useEffect(()=>{const el=ref.current;el?.showModal();return()=>el?.close()},[])
 return <dialog ref={ref} className={`modal ${wide?'wide':''}`} onCancel={e=>{e.preventDefault();onClose()}} onClick={e=>{if(e.target===e.currentTarget){const r=e.currentTarget.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)onClose()}}} aria-label={title}>
  <button className="icon-button modal-close" aria-label="Đóng cửa sổ" onClick={onClose}><X size={20}/></button>
  {eyebrow&&<p className="eyebrow">{eyebrow}</p>}<h2>{title}</h2>{children}
 </dialog>
}
