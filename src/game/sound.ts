let context:AudioContext|null=null
export function playSound(kind:'move'|'capture'|'check'|'end',volume:number){
 try{
  context??=new AudioContext();void context.resume()
  const notes=kind==='capture'?[190,110,310]:kind==='check'?[520,620]:kind==='end'?[330,440,550]:[300,210]
  notes.forEach((frequency,index)=>{const osc=context!.createOscillator(),gain=context!.createGain(),start=context!.currentTime+index*.055;osc.type=kind==='capture'?'triangle':'sine';osc.frequency.setValueAtTime(frequency,start);gain.gain.setValueAtTime(0,start);gain.gain.linearRampToValueAtTime(volume/100*.12,start+.012);gain.gain.exponentialRampToValueAtTime(.001,start+.16);osc.connect(gain);gain.connect(context!.destination);osc.start(start);osc.stop(start+.18)})
 }catch{/* Browsers without WebAudio still support the complete game. */}
}
