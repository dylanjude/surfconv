import numpy as np
import sys

if(len(sys.argv)>1):
    fin = sys.argv[1]
else:
    quit("Usage: ugrid2facet <file.ugrid>")

with open(fin, "r") as f:
    nv,n3f,n4f,n4,n5,n6,n8 = [int(xx) for xx in f.readline().split()]
    x = np.zeros((nv,3),'d')
    conn3 = np.zeros((n3f,3),'i')
    conn4 = np.zeros((n4f,4),'i')    
    for i in range(nv):
        x[i,:] = [float(xx) for xx in f.readline().split()]
    for i in range(n3f):
        conn3[i,:] = [int(xx) for xx in f.readline().split()]
    for i in range(n4f):
        conn4[i,:] = [int(xx) for xx in f.readline().split()]

fout = fin[:-5] + "facet"

with open(fout, "w") as f:
    f.write("FACET FILE V3.0    3-May-17   09:24:44   exported from Pointwise V17.1R4\n")
    f.write("1\n");
    f.write("Grid\n")
    f.write("0, 0.00 0.00 0.00 0.00\n")
    f.write("{:d}\n".format(nv))
    for i in range(nv):
        f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(x[i,0],x[i,1],x[i,2]))
    if(n4f>0):
        f.write("1\n")
        f.write("Quadrilaterals\n")
        f.write("{:d} 4\n".format(n4f))
        for e in range(n4f):
            for i in conn4[e,:]:
                f.write("{:d} ".format(i))
            f.write("0 0001 {:d}\n".format(e))
    if(n3f>0):
        f.write("1\n")
        f.write("Triangles\n")
        f.write("{:d} 3\n".format(n3f))
        for e in range(n3f):
            for i in conn3[e,:]:
                f.write("{:d} ".format(i))
            f.write("0 0001 {:d}\n".format(n4f+e))            
        
