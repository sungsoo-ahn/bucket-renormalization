import numpy as np

def get_logZ(ising_gm):
    grid_size = ising_gm.grid_size
    interaction = ising_gm.interaction
    edge_set = []
    for x in range(grid_size*grid_size):
        q, m = divmod(x, grid_size)
        if m != grid_size-1:
            edge_set.append([x,x+1])

        if q != grid_size-1:
            edge_set.append([x,x+grid_size])

    vcoupling = np.zeros((grid_size-1, grid_size))
    hcoupling = np.zeros((grid_size, grid_size-1))
    for i, e in enumerate(edge_set):
        var1, var2 = e
        if abs(var1 - var2) == 1:
            h_i, h_j = divmod(min(var1, var2), grid_size)
            hcoupling[h_i, h_j] = interaction[i]
        elif abs(var1 - var2) == grid_size:
            v_i, v_j = divmod(min(var1, var2), grid_size)
            vcoupling[v_i, v_j] = interaction[i]
    mat = np.zeros((4*grid_size*grid_size, 4*grid_size*grid_size))
    thcoupling = np.tanh(hcoupling)
    tvcoupling = np.tanh(vcoupling)
    for i in range(grid_size*grid_size):
        mat[4*i+1,4*i]=1
        mat[4*i,4*i+1]=-1
        mat[4*i,4*i+2]=1
        mat[4*i+2,4*i]=-1
        mat[4*i,4*i+3]=1
        mat[4*i+3,4*i]=-1
        mat[4*i+2,4*i+1]=1
        mat[4*i+1,4*i+2]=-1
        mat[4*i+3,4*i+1]=1
        mat[4*i+1,4*i+3]=-1
        mat[4*i+3,4*i+2]=1
        mat[4*i+2,4*i+3]=-1

    for i in range(grid_size):
        for j in range(grid_size-1):
            kv=grid_size*i+j+1
            kh=grid_size*j+i+1
            mat[4*kv,4*(kv-1)+3] = tvcoupling[j,i]
            mat[4*(kv-1)+3,4*kv] = -tvcoupling[j,i]
            mat[4*(kh-1)+2,4*(kh+grid_size-1)+1] = thcoupling[i,j]
            mat[4*(kh+grid_size-1)+1,4*(kh-1)+2] = -thcoupling[i,j]

    return grid_size*grid_size*np.log(2.0)+ \
                  np.sum(np.log(np.cosh(vcoupling)))+ \
                  np.sum(np.log(np.cosh(hcoupling)))+ \
                  0.5*np.log(abs(np.linalg.det(mat)))
