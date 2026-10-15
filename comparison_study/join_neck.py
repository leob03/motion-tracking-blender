"""Join different neck boundary loops with a fixed, oriented triangle collar.
Only the lower FLAME neck band is deformed and faired. Upper face is fixed.
"""
import numpy as np
from collections import defaultdict

def boundaries(faces):
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);_,inv,count=np.unique(np.sort(edges,1),axis=0,return_inverse=True,return_counts=True);edges=edges[count[inv]==1];nxt={int(a):int(b) for a,b in edges};assert len(nxt)==len(edges)
    remaining=set(nxt);loops=[]
    while remaining:
        first=min(remaining);loop=[first];remaining.remove(first);j=nxt[first]
        while j!=first:loop.append(j);remaining.remove(j);j=nxt[j]
        loops.append(np.array(loop))
    return loops

def join(body,body_faces,head,head_faces,rotation,protected_vertices):
    body_loop=max(boundaries(body_faces),key=len)[::-1]
    # FLAME also has a mouth boundary. The neck is the lowest loop in head-local Y.
    loops=boundaries(head_faces);neutral_local=head[0]@rotation[0].T
    head_loop=min(loops,key=lambda ids:neutral_local[ids,1].mean())
    # Fixed cyclic correspondence; loop winding already chosen to close both surfaces.
    A=body[0,body_loop];B=head[0,head_loop]
    def centered(x):
        x=x-x.mean(0);return x/max(np.linalg.norm(x,axis=1).mean(),1e-8)
    aa=centered(A);bb=centered(B);m=len(A);n=len(B);u=np.arange(n)*m/n;ii=u.astype(int);alpha=u-ii
    interp=aa[ii]*(1-alpha[:,None])+aa[(ii+1)%m]*alpha[:,None]
    shifts=[np.mean((interp-np.roll(bb,-k,axis=0))**2) for k in range(n)];head_loop=np.roll(head_loop,-int(np.argmin(shifts)))
    target=body[:,body_loop[ii]]*(1-alpha[None,:,None])+body[:,body_loop[(ii+1)%m]]*alpha[None,:,None]
    # A short upward collar separates rings; in head-local frame upward is +Y.
    target+=rotation[:,1,:][:,None]*.008
    displacement=target-head[:,head_loop];result=head.copy();result[:,head_loop]=target
    adj=defaultdict(set)
    for tri in head_faces:
        for a,b in zip(tri,np.roll(tri,-1)):adj[int(a)].add(int(b));adj[int(b)].add(int(a))
    owner={int(v):i for i,v in enumerate(head_loop)};distance={int(v):0 for v in head_loop};front=list(owner)
    for level in range(1,5):
        fresh=[]
        for v in front:
            for nb in sorted(adj[v]):
                if nb not in distance:distance[nb]=level;owner[nb]=owner[v];fresh.append(nb)
        front=fresh
    for v,level in distance.items():
        if level:result[:,v]+=displacement[:,owner[v]]*(.5+.5*np.cos(np.pi*level/4))
    # Zipper with proportional arc progress. Both existing boundary edges receive
    # exactly one opposite directed bridge edge, preserving manifold orientation.
    bridge=[];i=j=0;offset=body.shape[1]
    while i<m or j<n:
        ai=int(body_loop[i%m]);bj=int(head_loop[j%n])+offset
        if j==n or (i<m and (i+1)/m<=(j+1)/n):bridge.append([ai,int(body_loop[(i+1)%m]),bj]);i+=1
        else:bridge.append([ai,int(head_loop[(j+1)%n])+offset,bj]);j+=1
    faces=np.concatenate([body_faces,head_faces+offset,np.array(bridge)])
    # Fair the lower neck over a wider band. The first prototype's short band
    # folded when the wide FLAME collar was squeezed into the MHR neck hole.
    # Biharmonic fairing respects the fixed face and fixed MHR body, including
    # their neighboring surface tangents across the seam.
    from scipy import sparse
    from scipy.sparse.linalg import factorized
    result[:,protected_vertices]=head[:,protected_vertices]
    vertices=np.concatenate([body,result],1)
    local=head[0]@rotation[0].T
    band=np.flatnonzero(local[:,1]<local[head_loop,1].max()+.052)
    band=np.setdiff1d(band,protected_vertices);unknown=band+offset;N=vertices.shape[1]
    e=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);e=np.concatenate([e,e[:,::-1]])
    adj=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(N,N)).tocsr();adj.data[:]=1
    degree=np.asarray(adj.sum(1)).ravel();L=sparse.eye(N)-sparse.diags(1/np.maximum(degree,1))@adj
    U=L[:,unknown];penalty=.0005;solve=factorized((U.T@U+penalty*sparse.eye(len(unknown))).tocsc())
    for frame in range(len(vertices)):
        fixed=vertices[frame].copy();old=fixed[unknown].copy();fixed[unknown]=0
        vertices[frame,unknown]=solve(-U.T@(L@fixed)+penalty*old)
    result=vertices[:,offset:];changed=np.setdiff1d(np.union1d(band,np.array([v for v,d in distance.items() if d<4])),protected_vertices)
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);u,inv,c=np.unique(np.sort(edges,1),axis=0,return_inverse=True,return_counts=True)
    original_neck={tuple(sorted((int(a),int(b)))) for loop in [body_loop,head_loop+offset] for a,b in zip(loop,np.roll(loop,-1))};edge_counts={tuple(e):int(k) for e,k in zip(u,c)};assert all(edge_counts[e]==2 for e in original_neck);assert c.max()==2
    for edge in original_neck:
        directed=edges[(np.sort(edges,1)==edge).all(1)];assert np.array_equal(directed[0],directed[1,::-1])
    upper=np.setdiff1d(np.arange(head.shape[1]),changed);assert np.array_equal(result[:,upper],head[:,upper])
    assert np.array_equal(result[:,protected_vertices],head[:,protected_vertices])
    vertices=np.concatenate([body,result],1);tris=vertices[:,np.array(bridge)];area=np.linalg.norm(np.cross(tris[:,:,1]-tris[:,:,0],tris[:,:,2]-tris[:,:,0]),axis=-1)/2
    assert np.isfinite(vertices).all() and area.min()>1e-10
    return vertices,faces,dict(body_boundary_vertices=m,flame_neck_boundary_vertices=n,bridge_triangles=len(bridge),neck_edges_closed=True,consistent_winding=True,nonmanifold_edges=int((c>2).sum()),modified_lower_neck_vertices=len(changed),face_outside_neck_band_max_change_m=0,protected_contour_and_landmark_vertices=len(protected_vertices),protected_face_vertices_max_change_m=0,minimum_bridge_triangle_area_m2=float(area.min()),max_boundary_adjustment_m=float(np.linalg.norm(displacement,axis=-1).max()),remaining_open_boundary_edges=int((c==1).sum()),fairing='Biharmonic lower-neck surface fairing, MHR body and upper FLAME face fixed',note='Shared boundary indices join the head and body into one skin surface. FLAME mouth boundary remains open. No new armature/skinning or collision solve.')
