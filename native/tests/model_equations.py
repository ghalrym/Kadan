"""Independent bounded representative-shape prefix equations, stdlib only.
New test-only oracle requiring review. Never imports native/production inference,
reads captures, or derives expected values from the implementation under test.
"""
import math


def configure(g):
    """H=256, linear Kheads=2/Vheads=4, full D=8/R=4, E=12/top8."""
    bf,f,bits,norm,sig = (g[k] for k in ('bf','f','bits','norm','sigmoid'))
    H,KH,VH,KD,VD,D,E,K=256,2,4,4,4,8,12,8
    C=2*KH*KD+VH*VD
    # Small signed representable entries keep the wider dot products bounded.
    def matrix(rows,cols,role):
        return [[(-1)**(r+j+role)*.015625*(1+(2*r+j+role)%4) for j in range(cols)] for r in range(rows)]
    shapes={'lqkv':(C,H,0),'lz':(VH*VD,H,1),'lout':(H,VH*VD,2),
            'fqg':(4*D,H,5),'fk':(D,H,6),'fv':(D,H,7),'fout':(H,2*D,8)}
    g['base_matrices']={k:matrix(*v) for k,v in shapes.items()}
    g.update(embedding=[[.125*((3*r+5*j+r*j)%13-6) for j in range(H)] for r in range(16)],
             final_norm=[.03125*(j%7-3) for j in range(H)],
             innorm=[.125*(j%4-1) for j in range(H)],postnorm=[.0625*(j%5-2) for j in range(H)],
             la=matrix(VH,H,3),lb=matrix(VH,H,4),
             conv=[[.25,-.5,.75] if j%2 else [-.25,.5,1] for j in range(C)],
             logs=[-.5,.25,-.25,.5],dt=[.25,-.125,.125,-.25],gate=[.75,1.25,1,.5],
             qn=[.25,-.125,.5,0]*2,kn=[-.25,.125,0,.25]*2,
             router=[[1 if j==e else .03125*(e-5) if j==E else 0 for j in range(H)] for e in range(E)],
             shared_gate=[(-1)**j*.0078125*(1+j%3) for j in range(H)])
    g['head']=[[g['head_weight'](r,j) for j in range(H)] for r in range(16)]
    def dot(w,x):return g['dot'](w,x)
    def square_sum(row):
        s=0
        for x in row:s=f(s+f(x*x))
        return s
    def rotate(row,t):
        out=row.copy()
        for j in range(2):
            phase=f(t*f(1/10000**(2*j/4)))
            cs,sn=bf(math.cos(phase)),bf(math.sin(phase));a,b=row[j],row[j+2]
            out[j]=bf(bf(a*cs)-bf(b*sn));out[j+2]=bf(bf(a*sn)+bf(b*cs))
        return out
    def attention_full():
        keys=[];values=[];result=[]
        for t,token in enumerate(g['inputs']):
            x=norm(token,g['innorm']);qg=dot(g['fqg'],x)
            keys.append(rotate(norm(dot(g['fk'],x),g['kn']),t));values.append(dot(g['fv'],x));gated=[]
            for h in range(2):
                query=rotate(norm(qg[2*D*h:2*D*h+D],g['qn']),t)
                logits=[bf(bf(sum(a*b for a,b in zip(query,k)))*f(1/f(math.sqrt(D)))) for k in keys]
                exps=[math.exp(v-max(logits)) for v in logits];probs=[bf(e/sum(exps)) for e in exps]
                core=[bf(sum(p*v[j] for p,v in zip(probs,values))) for j in range(D)]
                gated.extend(bf(v*bf(sig(z))) for v,z in zip(core,qg[2*D*h+D:2*D*h+2*D]))
            result.append(([bf(a+b) for a,b in zip(token,dot(g['fout'],gated))],
                           [bits(v) for row in keys for v in row]+[0]*(D*(7-t)),
                           [bits(v) for row in values for v in row]+[0]*(D*(7-t))))
        return result
    def attention_linear():
        history=[[0]*3 for _ in range(C)];state=[[[0]*VD for _ in range(KD)] for _ in range(VH)];result=[]
        for token in g['inputs']:
            x=norm(token,g['innorm']);projected=dot(g['lqkv'],x);zed=dot(g['lz'],x);av=dot(g['la'],x);bv=dot(g['lb'],x);activated=[]
            for ch in range(C):
                history[ch]=history[ch][1:]+[projected[ch]];raw=dot([g['conv'][ch]],history[ch])[0]
                activated.append(bf(f(raw*f(sig(raw)))))
            queries=[];keys=[]
            for h in range(KH):
                for dest,base,isq in ((queries,KD*h,True),(keys,KH*KD+KD*h,False)):
                    row=activated[base:base+KD];inv=f(1/f(math.sqrt(f(square_sum(row)+f(1e-6)))))
                    dest.append([f(f(v*inv)/f(math.sqrt(KD))) if isq else f(v*inv) for v in row])
            core=[]
            for h in range(VH):
                t=f(av[h]+g['dt'][h]);soft=f(max(t,0)+f(math.log1p(f(math.exp(-abs(t))))))
                decay=f(math.exp(f(-f(math.exp(g['logs'][h]))*soft)));beta=bf(f(sig(bv[h])))
                key,query=keys[h//(VH//KH)],queries[h//(VH//KH)];old=state[h];v=activated[2*KH*KD+h*VD:2*KH*KD+(h+1)*VD]
                # Expanded transition matrix, not the native incremental delta loop.
                transform=[[decay*((i==j)-beta*key[i]*key[j]) for j in range(KD)] for i in range(KD)]
                state[h]=[[sum(transform[i][j]*old[j][col] for j in range(KD))+beta*key[i]*v[col] for col in range(VD)] for i in range(KD)]
                core.extend(bf(sum(query[i]*state[h][i][col] for i in range(KD))) for col in range(VD))
            gated=[]
            for h in range(VH):
                row=core[VD*h:VD*(h+1)];inv=f(1/f(math.sqrt(f(f(square_sum(row)/VD)+f(1e-6)))))
                gated.extend(bf(f(bf(f(bf(f(row[j]*inv))*g['gate'][j]))*f(zed[VD*h+j]*f(sig(zed[VD*h+j]))))) for j in range(VD))
            result.append(([bf(a+b) for a,b in zip(token,dot(g['lout'],gated))],
                           [bits(v) for row in history for v in row],[f(v) for head in state for row in head for v in row]))
        return result
    def expert(e,x):
        def mat(role):
            middle=32 if e==E else 16;rows,cols=(H,middle) if role==2 else (middle,H)
            def weight(r,j):
                code=(r+2*j+3*e+role)%5;sign=-1 if (r+j+e+role)%3==0 else 1
                scale=f(((3*e+role)%4+2)/4*g['moe_scale'])
                return f(sign*[0,.5,1,1.5,2][code]*(.25 if (r+j//16+e+role)%2 else .125)*scale)
            return [[weight(r,j) for j in range(cols)] for r in range(rows)]
        gate,up=dot(mat(0),x),dot(mat(1),x)
        return dot(mat(2),[bf(bf(a*sig(a))*b) for a,b in zip(gate,up)])
    def mixture(x):
        logits=dot(g['router'],x);exps=[math.exp(v-max(logits)) for v in logits];prob=[v/sum(exps) for v in exps]
        ids=sorted(range(E),key=lambda e:(-prob[e],e))[:K];weights={e:bf(prob[e]/sum(prob[j] for j in ids)) for e in ids}
        routed=[0]*H
        for e in sorted(ids):routed=[bf(a+bf(b*weights[e])) for a,b in zip(routed,expert(e,x))]
        factor=bf(sig(dot([g['shared_gate']],x)[0]))
        return [bf(a+bf(b*factor)) for a,b in zip(routed,expert(E,x))]
    g.update(attention_full=attention_full,attention_linear=attention_linear,mixture=mixture)
