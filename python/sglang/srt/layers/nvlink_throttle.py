"""Low-overhead software-equivalent NVLink link-count limiter.

Physical links are unchanged.  The limiter accumulates serialization debt and
only sleeps after a configurable threshold, avoiding sub-microsecond sleeps on
every A/F message.  Audit data is aggregated and flushed infrequently.
"""
from __future__ import annotations
import atexit,json,os,threading,time
from pathlib import Path
_LOCK=threading.Lock();_next_available=0.0;LINKS_NATIVE=8
_STATS={'messages':0,'bytes':0,'added_serialization_s':0.0,'sleep_s':0.0,'limiter_wall_s':0.0};_LAST_FLUSH=0

def configured_links()->int:
 value=int(os.environ.get('AFD_NVLINK_EQUIVALENT_LINKS','8'))
 if value not in (1,2,4,8):raise ValueError('AFD_NVLINK_EQUIVALENT_LINKS must be one of 1,2,4,8')
 return value

def bandwidth_GBs()->float:
 override=os.environ.get('AFD_NVLINK_BANDWIDTH_GBS')
 return float(override) if override else configured_links()*float(os.environ.get('AFD_NVLINK_PER_LINK_GBS','25'))
def _flush():
 path=os.environ.get('AFD_NVLINK_AUDIT_LOG')
 if not path or not _STATS['messages']:return
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 row={'t':time.time(),'pid':os.getpid(),'equivalent_links':configured_links(),'target_bandwidth_GBs':bandwidth_GBs(),'native_bandwidth_GBs':LINKS_NATIVE*float(os.environ.get('AFD_NVLINK_PER_LINK_GBS','25')),**_STATS}
 with open(path,'a') as f:f.write(json.dumps(row)+'\n')
atexit.register(_flush)

def throttle_and_audit(tensor,direction='send'):
 global _next_available,_LAST_FLUSH
 nbytes=int(tensor.numel()*tensor.element_size());links=configured_links();bw=bandwidth_GBs();native_bw=LINKS_NATIVE*float(os.environ.get('AFD_NVLINK_PER_LINK_GBS','25'));added=max(0.0,nbytes/(bw*1e9)-nbytes/(native_bw*1e9));sleep_s=0.0;started=time.perf_counter()
 if links<LINKS_NATIVE and nbytes:
  threshold=float(os.environ.get('AFD_NVLINK_SLEEP_THRESHOLD_US','50'))/1e6
  with _LOCK:
   now=time.perf_counter();_next_available=max(now,_next_available)+added;debt=_next_available-now
   if debt>=threshold:sleep_s=debt;_next_available=now
  if sleep_s:time.sleep(sleep_s)
 elapsed=time.perf_counter()-started
 with _LOCK:
  _STATS['messages']+=1;_STATS['bytes']+=nbytes;_STATS['added_serialization_s']+=added;_STATS['sleep_s']+=sleep_s;_STATS['limiter_wall_s']+=elapsed
  flush_every=int(os.environ.get('AFD_NVLINK_AUDIT_FLUSH_MESSAGES','65536'))
  if _STATS['messages']-_LAST_FLUSH>=flush_every:_flush();_LAST_FLUSH=_STATS['messages']
 return {'bytes':nbytes,'sleep_s':sleep_s,'wall_s':elapsed,'equivalent_links':links,'target_bandwidth_GBs':bw}
