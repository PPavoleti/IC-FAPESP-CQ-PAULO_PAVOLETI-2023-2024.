import matplotlib.pyplot as plt
import numpy as np
import cmath
import math
from numpy import linalg as la
from numpy.linalg import norm
from qiskit import *

def C(M,vec,n,t):
    aux=(norm(vec)*(t)**n)/np.math.factorial(n)
    return aux

def D(M,vec,n,t):
    aux=(norm(vec)*(t)**(n-1))*t/np.math.factorial(n)
    return aux

def CG(num,vec,n,t):
    aux=(norm(vec)*((num*t)**n))/np.math.factorial(n)
    return aux

def DG(num,vec,n,t):
    aux=(norm(vec)*((num*t)**(n-1))*t)/np.math.factorial(n)
    return aux

def brandr(dig,num):
    """
    Generates a random binary number
    """
    a=[int(x) for x in list('{0:0b}'.format(num))]
    for j in range(dig-len(a)):
        b=[0]+a
        a=b
    return a

def zeros(M):
    """
    Creates a zeros matrix
    """
    aux1=[0.0]
    for i in range(M-1):
        aux1=aux1+[0.0]
        aux2=[aux1]
    for i in range (M-1):
        aux3=np.concatenate((aux2,[aux1]), axis=0)
        aux2=aux3
    return aux3

def noseq(n,m):
	"""
	Creates an unordered sequrency
	"""
    v=[n]
    for i in range(m-1):
        v=v+[n-i-1]
    return v

def oseq(init,fin):
	"""
	Creats an ordered sequency
	"""
    v=[init]
    n=fin-init
    for i in range(n):
        v=v+[init+i+1]
    return v

def summat(m1,m2):
    dim=len(m1)
    s=mat_zero(dim)
    for i in range(dim):
        for j in range (dim):
            s[i][j]=m1[i][j]+m2[i][j]
    return s

def ang(vec):
    """
    Evaluate for the rotation matrixes
    to be used with the controlled operators.
    """
    dim=len(vec)
    n=int(np.log2(dim))
    y=zeros(dim)
    z=zeros(dim)
    c=zeros(dim)
    a=zeros(dim)
    nums=[0]*dim

    for m in range(dim):
        c[n][m]=abs(vec[m])
        a[n][m]=cmath.phase(vec[m])
    auxn=[0]*n
    for m in range(n):
        auxn[m]=n-m-1
    
    for aux in auxn:
        for j in range(pow(2,aux)):
            k=j*2
            cateto1=c[aux+1][k]
            cateto2=c[aux+1][k+1]
            hip=np.sqrt(pow(cateto1,2)+pow(cateto2,2))
            formatted_string = "{:.6f}".format(hip)
            var=float(formatted_string)
            if (var!=0.000000):
                y[aux][j]=2*np.arctan2(cateto2,cateto1)
            else:
                y[aux][j]=0.0
            c[aux][j]=hip

            if(aux!=0):
                z[aux][j]=-a[aux+1][k]+a[aux+1][k+1]
                a[aux][j]=a[aux+1][k]+0.5*z[aux][j]
            else:
                z[aux][j]=-2*a[aux+1][k]
                s=a[aux+1][k+1]-0.5*z[aux][j]
    return dim,n,s,y,z

def prep(vec):
	"""
	Does the encoding part to be used
	with the Tao Xin method.
	"""
    tam,qbits,shift,ang_y,ang_z=ang(vec)
    ycircuit=QuantumCircuit(qbits)
    zcircuit=QuantumCircuit(qbits)
    ycircuit.ry(ang_y[0][0],qbits-1)
    zcircuit.rz(ang_z[0][0],qbits-1)
    indices=[0]*(qbits-1)
    for m in range(qbits-1):
        indices[m]=m+1
    for indice in indices:
        for j in range(pow(2,indice)):
            qcz=QuantumCircuit(1)
            qcy=QuantumCircuit(1)
            qcy.ry(ang_y[indice][j],0)
            qcz.rz(ang_z[indice][j],0)
            k=j*pow(2,qbits-indice)
            vec=brandr(qbits,k)
            for i in range(indice):
                if(vec[i]==0):
                    ycircuit.x(qbits-i-1)
                    zcircuit.x(qbits-i-1)

            con_y=qcy.to_gate().control(indice)
            con_z=qcz.to_gate().control(indice)
            ycircuit.append(con_y,noseq(qbits-1,indice)+[qbits-1-indice])
            zcircuit.append(con_z,noseq(qbits-1,indice)+[qbits-1-indice])
            for i in range(indice):
                if(vec[i]==0):
                    ycircuit.x(qbits-i-1)
                    zcircuit.x(qbits-i-1)
    cirprep=QuantumCircuit(qbits)
    cirprep=cirprep.compose(ycircuit,range(qbits))
    cirprep.p(shift,qbits-1)
    cirprep=cirprep.compose(zcircuit,range(qbits))
    return cirprep

def txun(M,b,x0,t,k):
	"""
	Tao Xin algorithm to solve LDEs system.
	(Unitary case)
	"""
    N=len(x0)
    n=math.ceil(np.log2(N))
    anc=math.ceil(np.log2(k+1))
    Cnum=0.0
    Dnum=0.0
    vc=[0]*(2**anc)
    vd=[0]*(2**anc)
    aux=0.0
    
    for i in range(k+1):
        aux=aux+C(M,x0,i,t)
    Cnum=np.sqrt(aux)
    aux=0.0
    for i in range(k):
        aux=aux+D(M,b,i+1,t)
    Dnum=np.sqrt(aux)
    alep=np.sqrt(Cnum**2+Dnum**2)
    
    for i in range(k+1):
        vc[i]=np.sqrt(C(M,x0,i,t))/Cnum
    for i in range(k):
        vd[i]=np.sqrt(D(M,b,i+1,t))/Dnum
    
    v=[[Cnum/alep,Dnum/alep],[Dnum/alep,-Cnum/alep]]
    circ=QuantumCircuit(n+anc+1)
    circ.unitary(v,n+anc)

    conUx=prep(x0).to_gate().control(1)
    conb=prep(b).to_gate().control(1)
    circ.x(n+anc)
    circ.append(conUx,[n+anc]+oseq(0,n-1))
    circ.x(n+anc)
    circ.append(conb,[n+anc]+oseq(0,n-1))
    
    conVs1=prep(vc).to_gate().control(1)
    conVs2=prep(vd).to_gate().control(1)
    circ.x(n+anc)
    circ.append(conVs1,[n+anc]+oseq(n,n+anc-1))
    circ.x(n+anc)
    circ.append(conVs2,[n+anc]+oseq(n,n+anc-1))
    
    circ1=QuantumCircuit(n+anc+1)
    A=M
    for m in range(anc):
        auxc=QuantumCircuit(n)
        mpow=la.matrix_power(A,2**m)
        auxc.unitary(mpow,oseq(0,n-1))
        conUm=auxc.to_gate().control(1)
        circ1.append(conUm,[n+m]+oseq(0,n-1))
    
    conVs1dag=prep(vc).to_gate().inverse().control(1)
    conVs2dag=prep(vd).to_gate().inverse().control(1)
    circ1.x(n+anc)
    circ1.append(conVs1dag,[n+anc]+oseq(n,n+anc-1))
    circ1.x(n+anc)
    circ1.append(conVs2dag,[n+anc]+oseq(n,n+anc-1))
    circ1.unitary(v,n+anc).inverse()
    circ=circ.compose(circ1,range(n+anc+1))
    return alep,circ

def txm(M,mats,coefs,b,x_0,t,k)
	"""
	Tao Xin algorithm to solve LDEs system.
	(Includes particular solutions)
	"""
    n=math.ceil(np.log2(len(x_0)))
    L=len(coefs)
    qdits=math.ceil(np.log2(L))
    arr=[0]*L
    soma=0.0
    for i in range(L):
        soma=soma+coefs[i]
    fator=soma
    #Calculo das constantes g1,g2
    aux=0.0
    for i in range(L):
        aux=aux+coefs[i]
    g1=0.0
    for i in range (k+1):
        g1=g1+CG(fator,x_0,i,t)
    g1=np.sqrt(g1)
    g2=0.0
    for i in range(k):
        g2=g2+DG(fator,b,i+1,t)
    g2=np.sqrt(g2)
    S=g1**2+g2**2
    
    c=np.sqrt(g1**2+g2**2)
    v=[[g1/c,g2/c],[g2/c,-g1/c]]
    reg=1+k+k*qdits+n
    circ=QuantumCircuit(reg)
    circ.unitary(v,reg-1)
    
    vecV=[0]*(2**qdits)
    for i in range(L):
        vecV[i]=np.sqrt(coefs[i])
    gate=prep(vecV).to_gate()
    for i in range(k):
        init=n+i*qdits
        fin=init+qdits-1
        circ.append(gate,ord_seq(init,fin))
    
    conUx=prep(x_0).to_gate().control(1)
    conb=prep(b).to_gate().control(1)
    circ.x(reg-1)
    circ.append(conUx,[reg-1]+ord_seq(0,n-1))
    circ.x(reg-1)
    circ.append(conb,[reg-1]+ord_seq(0,n-1))
    circ.barrier()
    
    vecs1=[0]*(2**k)
    vecs2=[0]*(2**k)
    for j in range(k+1):
        ind=2**k-2**(k-j)
        vecs1[ind]=np.sqrt(CG(fator,x_0,j,t))
    for j in range (k):
        ind=2**k-2**(k-j)
        vecs2[ind]=np.sqrt(DG(fator,b,j+1,t))
    a1=norm(vecs1)
    a2=norm(vecs2)
    for j in range(2**k):
        vecs1[j]=vecs1[j]/a1
        vecs2[j]=vecs2[j]/a2
    convs1=prep(vecs1).to_gate().control(1)
    convs2=prep(vecs2).to_gate().control(1)
    circ.x(reg-1)
    circ.append(convs1,[reg-1]+ord_seq(n+k*qdits,n+k*qdits+k-1))
    circ.x(reg-1)
    circ.append(convs2,[reg-1]+ord_seq(n+k*qdits,n+k*qdits+k-1))
    circ.barrier()
    
    for i in range(k):
        for j in range(L):
            vec=binr(qdits,j)
            auxc=QuantumCircuit(n)
            auxc.unitary(mats[j],ord_seq(0,n-1))
            gate=auxc.to_gate().control(qdits+1)
            fin=reg-k-2-i*qdits
            init=fin-qdits+1
            for h in range(qdits):
                if(vec[h]==0):
                    circ.x(init+h)
            circ.append(gate,[reg-2-i]+ord_seq(fin-qdits+1,fin)+[0,n-1])
    for h in range(qdits):
        if(vec[h]==0):
            circ.x(init+h)
        circ.barrier()
    
    convs1dag=prep(vecs1).to_gate().inverse().control(1)
    convs2dag=prep(vecs2).to_gate().inverse().control(1)
    circ.x(reg-1)
    circ.append(convs1dag,[reg-1]+ord_seq(n+k*qdits,n+k*qdits+k-1))
    circ.x(reg-1)
    circ.append(convs2dag,[reg-1]+ord_seq(n+k*qdits,n+k*qdits+k-1))
    circ.barrier()
    
    gatedag=prep(vecV).to_gate().inverse()
    for i in range(k):
        init=n+i*qdits
        fin=init+qdits-1
        circ.append(gatedag,ord_seq(init,fin))
    
    circ.unitary(v,reg-1).inverse()
    return S,circ

def getsol(vec, anc, num):
	"""
	Does the decoding used to get the
	output from Tao Xin algorithms.
	"""
    sol=[0]
    tam=len(vec)
    dig=math.ceil(np.log2(tam))
    for i in range(tam):
        alg=binr(dig,i)
        aux=0
    for j in range(anc):
        aux=aux+alg[j]
    if (aux==0):
        sol=sol+[vec[i]*num]
    return sol
