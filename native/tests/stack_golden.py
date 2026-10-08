"""Independent stdlib mixed-stack equations; no production/native/model imports.
Frozen prefix-matrix full attention and expanded matrix linear recurrence.
Generate: python3 native/tests/stack_golden.py > native/tests/stack_golden.hpp
New test-only generator: requires separate user review before merge.
"""
import math
import struct

def f(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):
    u=struct.unpack('<I',struct.pack('<f',x))[0]
    return (u+0x7fff+((u>>16)&1))>>16

def bf(x):return struct.unpack('<f',struct.pack('<I',bits(x)<<16))[0]
def sigmoid(x):return 1/(1+math.exp(-x))
def norm(x,w):
    s=0
    for v in x:s=f(s+f(v*v))
    inv=f(1/f(math.sqrt(f(f(s/len(x))+f(1e-6)))))
    return [bf(f(f(v*inv)*f(1+z))) for v,z in zip(x,w)]
def dot(matrix,x):return [bf(sum(a*b for a,b in zip(row,x))) for row in matrix]
def matrix(rows,columns,role):
    return [[(-1)**(r+j+role)*.0625*(1+(2*r+j+role)%4) for j in range(columns)] for r in range(rows)]
def encode(x):
    for byte in range(256):
        exp,mant=(byte>>3)&15,byte&7
        if exp==15 and mant==7:continue
        val=(-1 if byte&128 else 1)*(mant*2**-9 if exp==0 else (1+mant/8)*2**(exp-7))
        if val==x:return byte
    raise ValueError(x)
inputs=[[(-1)**(j+t)*.25*(1+(j+2*t)%5) for j in range(16)] for t in range(3)]
innorm=[.125*(j%4-1) for j in range(16)]
postnorm=[.0625*(j%5-2) for j in range(16)]
lqkv,lz,lout=matrix(12,16,0),matrix(4,16,1),matrix(16,4,2)
la,lb=matrix(2,16,3),matrix(2,16,4)
conv=[[.25,-.5,.75] if j%2 else [-.25,.5,1] for j in range(12)]
logs=[-.5,.25];dt=[.25,-.125];gate=[.75,1.25]
fqg,fk,fv,fout=matrix(16,16,5),matrix(4,16,6),matrix(4,16,7),matrix(16,8,8)
qn=[.25,-.125,.5,0];kn=[-.25,.125,0,.25]
def rotate(x,t):
    x=x.copy();a,b=x[:2];cs,sn=bf(math.cos(t)),bf(math.sin(t));x[0]=bf(bf(a*cs)-bf(b*sn));x[1]=bf(bf(a*sn)+bf(b*cs));return x

def attention_full():
    keys=[];values=[];res=[]
    for t,token in enumerate(inputs):
        x=norm(token,innorm);qg=dot(fqg,x);keys.append(rotate(norm(dot(fk,x),kn),t));values.append(dot(fv,x));gated=[]
        for h in range(2):
            query=rotate(norm(qg[8*h:8*h+4],qn),t)
            logits=[bf(bf(sum(a*b for a,b in zip(query,k)))*.5) for k in keys]
            exps=[math.exp(x-max(logits)) for x in logits];probs=[bf(e/sum(exps)) for e in exps]
            core=[bf(sum(p*v[j] for p,v in zip(probs,values))) for j in range(4)]
            gated.extend(bf(v*bf(sigmoid(g))) for v,g in zip(core,qg[8*h+4:8*h+8]))
        res.append(([bf(a+b) for a,b in zip(token,dot(fout,gated))], [bits(x) for row in keys for x in row]+[0]*(4*(7-t)),[bits(x) for row in values for x in row]+[0]*(4*(7-t))))
    return res

def attention_linear():
    history=[[0]*3 for _ in range(12)];state=[[[0]*2 for _ in range(2)] for _ in range(2)];res=[]
    for token in inputs:
        x=norm(token,innorm);projected=dot(lqkv,x);zed=dot(lz,x);av=dot(la,x);bv=dot(lb,x);activated=[]
        for ch in range(12):
            history[ch]=history[ch][1:]+[projected[ch]];raw=dot([conv[ch]],history[ch])[0];activated.append(bf(f(raw*f(sigmoid(raw)))))
        queries=[];keys=[]
        for h in range(2):
            for dest,base,isq in [(queries,2*h,True),(keys,4+2*h,False)]:
                pair=activated[base:base+2];ss=f(f(pair[0]*pair[0])+f(pair[1]*pair[1]));inv=f(1/f(math.sqrt(f(ss+f(1e-6)))))
                dest.append([f(f(v*inv)/f(math.sqrt(2))) if isq else f(v*inv) for v in pair])
        core=[]
        for h in range(2):
            t=f(av[h]+dt[h]);soft=f(max(t,0)+f(math.log1p(f(math.exp(-abs(t))))));decay=f(math.exp(f(-f(math.exp(logs[h]))*soft)));beta=bf(f(sigmoid(bv[h])))
            key,query=keys[h],queries[h];old=state[h];v=activated[8+2*h:10+2*h]
            transform=[[decay*((i==j)-beta*key[i]*key[j]) for j in range(2)] for i in range(2)]
            state[h]=[[sum(transform[i][j]*old[j][col] for j in range(2))+beta*key[i]*v[col] for col in range(2)] for i in range(2)]
            core.extend(bf(sum(query[i]*state[h][i][col] for i in range(2))) for col in range(2))
        gated=[]
        for h in range(2):
            row=core[2*h:2*h+2];ss=f(f(row[0]*row[0])+f(row[1]*row[1]));inv=f(1/f(math.sqrt(f(f(ss/2)+f(1e-6)))))
            gated.extend(bf(f(bf(f(bf(f(row[j]*inv))*gate[j]))*f(zed[2*h+j]*f(sigmoid(zed[2*h+j]))))) for j in range(2))
        res.append(([bf(a+b) for a,b in zip(token,dot(lout,gated))],[bits(x) for row in history for x in row],[f(x) for head in state for row in head for x in row]))
    return res

def expert(e,x):
    def mat(role):
        middle=32 if e==4 else 16;rows,cols=(16,middle) if role==2 else (middle,16)
        def weight(r,j):
            nib=(r+2*j+3*e+role)%5;sign=-1 if (r+j+e+role)%3==0 else 1
            return sign*[0,.5,1,1.5,2][nib]*(.25 if (r+j//16+e+role)%2 else .125)*((3*e+role)%4+2)/4*moe_scale
        return [[weight(r,j) for j in range(cols)] for r in range(rows)]
    g,u=dot(mat(0),x),dot(mat(1),x)
    return dot(mat(2),[bf(bf(a*sigmoid(a))*b) for a,b in zip(g,u)])
router=[[1 if j==e else .125*(e-1) if j==4 else 0 for j in range(16)] for e in range(4)]
shared_gate=[(-1)**j*.03125*(1+j%3) for j in range(16)]
def mixture(x):
    logits=dot(router,x);exps=[math.exp(v-max(logits)) for v in logits];prob=[v/sum(exps) for v in exps];ids=sorted(range(4),key=lambda e:(-prob[e],e))[:2];ws=[bf(prob[e]/sum(prob[j] for j in ids)) for e in ids]
    routed=[0]*16
    for e in sorted(ids):routed=[bf(a+bf(b*ws[ids.index(e)])) for a,b in zip(routed,expert(e,x))]
    factor=bf(sigmoid(dot([shared_gate],x)[0]));result=[bf(a+bf(b*factor)) for a,b in zip(routed,expert(4,x))]
    return result

# Each prefix is recomputed from immutable token IDs. The generator does not
# invoke the incremental native coordinator or use its outputs as expectations.
base_matrices={name:globals()[name] for name in ['lqkv','lz','lout','fqg','fk','fv','fout']}
embedding=[[.125*((3*r+5*j+r*j)%13-6) for j in range(16)] for r in range(16)]
final_norm=[.03125*(j%7-3) for j in range(16)]
def head_code(r,j):
    r=0 if r==1 else r
    return (2*r+3*j+r*j)%5+(8 if (r+2*j)%3==0 else 0)
def head_weight(r,j):
    r=0 if r==1 else r;n=head_code(r,j)
    return (-1 if n&8 else 1)*[0,.5,1,1.5,2][n&7]*(.125 if r%2==0 else .25)*.75
head=[[head_weight(r,j) for j in range(16)] for r in range(16)]
def evaluate(token_ids):
    global inputs,moe_scale
    inputs=[embedding[t] for t in token_ids];records=[]
    for layer in range(4):
        scale=.5+.25*layer;moe_scale=.5+.125*layer
        for name,original in base_matrices.items():globals()[name]=[[v*scale for v in row] for row in original]
        attn=attention_full() if layer==3 else attention_linear();normalized=[norm(row[0],postnorm) for row in attn];mixed=[mixture(row) for row in normalized]
        inputs=[[bf(a+b) for a,b in zip(row[0],m)] for row,m in zip(attn,mixed)]
        records.append((inputs[-1],attn[-1][1],attn[-1][2]))
    normalized=norm(inputs[-1],final_norm);logits=dot(head,normalized);selected=max(range(16),key=lambda t:(logits[t],-t))
    return records,normalized,logits,selected
ids=[2,7];results=[]
results.append(evaluate(ids[:1]));results.append(evaluate(ids))
for _ in range(3):
    ids.append(results[-1][3]);results.append(evaluate(ids))
eos=next(i for i in reversed(range(16)) if i not in [r[3] for r in results])
print('// Generated by stack_golden.py; independent whole-prefix equations.\n#pragma once\n#include <array>\n#include <cstdint>\nnamespace stack_golden {')
def emit(name,values,typ='float'):
    def num(x):return float(x).hex()+'f' if typ=='float' else str(x)
    if isinstance(values[0],list):
        print(f'inline constexpr std::array<std::array<{typ},{len(values[0])}>,{len(values)}> {name}{{{{')
        for row in values:print('    {{'+','.join(num(x) for x in row)+'}},')
        print('}};')
    else:print(f'inline constexpr std::array<{typ},{len(values)}> {name}{{'+','.join(num(x) for x in values)+'};')
emit('input_ids',ids,'unsigned');emit('selected',[r[3] for r in results],'unsigned');print(f'inline constexpr unsigned eos={eos};')
emit('embedding',sum(embedding,[]));emit('final_norm',final_norm)
emit('head_packed',[head_code(r,j)|(head_code(r,j+1)<<4) for r in range(16) for j in range(0,16,2)],'std::uint8_t')
emit('head_scales',[0x20 if (0 if r==1 else r)%2==0 else 0x28 for r in range(16)],'std::uint8_t')
for layer in range(4):
    emit(f'layer{layer}_output',[r[0][layer][0] for r in results]);emit(f'layer{layer}_first',[r[0][layer][1] for r in results],'std::uint16_t');emit(f'layer{layer}_second',[r[0][layer][2] for r in results],'float' if layer<3 else 'std::uint16_t')
emit('normalized',[r[1] for r in results]);emit('logits',[r[2] for r in results]);print('}')
