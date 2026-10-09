"""Request-scoped fixed-shape block hooks: independent eager prefill, shard once, gather once per step."""
import torch
import torch.distributed as dist

from api.inference.image.parallel import TokenShard, cached_block, compact_prefix


class UlyssesAdapter:
    def __init__(self,transformer,rank,control,request_id):
        self.request_id=request_id
        self.transformer=transformer;self.rank=rank;self.control=control
        self.shard=TokenShard(16384,rank,2);self.mode=None;self.step=-1;self.prefix={};self.original=[]
    def install(self):
        for index,block in enumerate(self.transformer.transformer_blocks):
            original=block.forward;self.original.append((block,original))
            def forward(*args,_index=index,_block=block,_original=original,**kwargs):
                if self.mode!='cached':return _original(*args,**kwargs)
                if args or kwargs['kv_cache_mode']!='cached':raise ValueError('Unexpected cached block calling convention')
                hidden=kwargs['hidden_states']
                if _index==0:hidden=self.shard.take(hidden)
                if hidden.shape!=(1,8192,4096) or not bool(kwargs['target_token_mask'].all()):raise ValueError('Wrong local target ownership')
                if _index not in self.prefix:
                    self.prefix[_index]=compact_prefix(*kwargs['layer_cache'].get(),'ulysses',self.rank,2)
                mask=kwargs['attention_mask']
                if mask is not None and (mask.dtype!=torch.bool or mask.shape[1:3]!=(1,1)):raise ValueError('Unexpected production mask')
                return cached_block(_block,hidden,kwargs['modulation'],self.shard.take(kwargs['rotary_emb'],dim=0),
                    self.prefix[_index],self.shard,mode='ulysses',key_valid=None if mask is None else mask[:,0,0],
                    control_group=self.control,request_id=self.request_id,step=self.step,block_index=_index)
            block.forward=forward
        norm=self.transformer.norm_out;original=norm.forward;self.original.append((norm,original))
        def norm_forward(hidden,temb,mask):
            if self.mode=='cached':mask=self.shard.take(mask,dim=0)
            return original(hidden,temb,mask)
        norm.forward=norm_forward
    def gather(self,output):
        if self.mode!='cached':return output
        if output.shape!=(1,8192,64):raise ValueError('Wrong local final projection shape')
        parts=[torch.empty_like(output) for _ in range(2)]
        dist.all_gather(parts,output.contiguous())
        return torch.cat(parts,dim=1)
    def close(self):
        for module,forward in self.original:module.forward=forward
        self.prefix.clear()


def validate_transformer_output(output,mode):
    if output.ndim!=3 or output.shape[0]!=1 or output.shape[2]!=64:
        raise ValueError('Unexpected transformer output batch/channel shape')
    if mode=='extract':
        if output.shape[1]<16384:raise ValueError('Prefill output omits target rows')
    elif mode=='cached':
        if output.shape[1]!=16384:raise ValueError('Cached output must contain target rows only')
    else:raise ValueError('Unexpected cache mode')
    # The pinned pipeline owns tail slicing after prefill; never slice here.
    return output
