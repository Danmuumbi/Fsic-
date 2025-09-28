
import random, numpy as np
def build_rect(rows=10, cols=10, Jt=0.9, Jr=0.7, rewire_p=0.0, seed=0):
    random.seed(seed); links=[]
    for i in range(rows):
        for j in range(cols-1): links.append((i,j,i,j+1,'t',Jt))
    for i in range(rows-1):
        for j in range(cols): links.append((i,j,i+1,j,'r',Jr))
    if rewire_p>0.0:
        tang=[k for k,e in enumerate(links) if e[4]=='t']
        for k in tang:
            if random.random()<rewire_p:
                i,j,_,_,t,w = links[k]
                ii=random.randrange(rows); jj=random.randrange(cols)
                links[k]=(i,j,ii,jj,t,w)
    return {'rows':rows,'cols':cols,'links':links,'tiling':'rect'}

def build_hex(rows=10, cols=10, Jt=0.9, Jr=0.7, rewire_p=0.0, seed=0):
    random.seed(seed); links=[]
    for i in range(rows):
        for j in range(cols-1): links.append((i,j,i,j+1,'t',Jt))
    for i in range(rows-1):
        for j in range(cols):
            if i%2==0:
                if j-1>=0: links.append((i,j,i+1,j-1,'r',Jr))
                links.append((i,j,i+1,j,'r',Jr))
            else:
                links.append((i,j,i+1,j,'r',Jr))
                if j+1<cols: links.append((i,j,i+1,j+1,'r',Jr))
    if rewire_p>0.0:
        tang=[k for k,e in enumerate(links) if e[4]=='t']
        for k in tang:
            if random.random()<rewire_p:
                i,j,_,_,t,w = links[k]
                ii=random.randrange(rows); jj=random.randrange(cols)
                links[k]=(i,j,ii,jj,t,w)
    return {'rows':rows,'cols':cols,'links':links,'tiling':'hex'}

def to_mats(mask):
    rows, cols = mask['rows'], mask['cols']; N=rows*cols
    def idx(i,j): return i*cols+j
    Jt=np.zeros((N,N)); Jr=np.zeros((N,N))
    for i,j,k,l,t,w in mask['links']:
        a=idx(i,j); b=idx(k,l)
        if t=='t': Jt[a,b]+=w; Jt[b,a]+=w
        else:      Jr[a,b]+=w; Jr[b,a]+=w
    return Jt, Jr, N
