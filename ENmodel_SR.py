import numpy as np
import matplotlib.pyplot as pl

def HSVtoRGB(h,s,v):
    '''
    HSV colour wheel, supply inputs in range 0,1
    '''
    i = np.floor(h*6)
    f = h*6.-i
    p = v*(1.-s)
    q = v*(1.-f*s)
    t = v*(1.-(1.-f)*s)
    imod = i%6

    if(imod==0): return p,q,v
    elif(imod==1): return t,p,v
    elif(imod==2): return v,p,q
    elif(imod==3): return v,t,p
    elif(imod==4): return q,v,p
    elif(imod==5): return p,v,t

def step(E,x,h=0.05):
    '''
    One iteration of dynamics (4th-order Runge Kutta)
    '''

    h2 = h*h
    k1 = h*(E+eta*laplace(x))
    k2 = 0.5*h2*(E+eta*laplace(x+k1*0.5))
    k3 = 0.5*h2*(E+eta*laplace(x+k2*0.5))
    k4 = h2*(E+eta*laplace(x+k3))
    return x+(k1+2.0*(k2+k3)+k4)/6.0

def laplace(X):
    '''
    2D laplacian with periodic boundary conditions
    '''

    n = X.shape[0]
    Y = np.array(np.zeros([n+2,n+2]),dtype=X.dtype)
    Y[1:n+1,1:n+1] = X
    Y[0,1:n+1] = X[n-1,:]
    Y[n+1,1:n+1] = X[0,:]
    Y[1:n+1,0]  = X[:,n-1]
    Y[1:n+1,n+1] = X[:,0]

    return (Y[2:n+2,1:n+1]+Y[0:n,1:n+1]+Y[1:n+1,2:n+2]+Y[1:n+1,0:n]-4.*Y[1:n+1,1:n+1])

# Threshold (<1) for identifying pinwheels

thresh = 0.5

# Model params
eta = 0.06  # strength of lateral interactions
sig = 0.5   # afferent acivation selectivity

oneOverTwoSigmaSquared = 1./(2.*sig**2)

# Simulation params

N = 101     # (sqrt) number of units (note fft assumes an odd number is used)
T = 12000    # iterations



# Initial (random) orientation/retinotopic maps

z = np.exp(2.j*np.random.rand(N,N)*np.pi)
r = np.array(np.zeros([N,N]),dtype=complex)

for i in range(N):
    for j in range(N):
        r[i][j] = np.complex((np.random.rand()-0.5)*0.01,(np.random.rand()-0.5)*0.01)

# Main simulation loop

for t in range(T):

    # Generate input pattern

    sr = np.complex(np.random.rand()-0.5,np.random.rand()-0.5)
    sz = np.exp(2.j*np.random.rand())

    # Compute afferent response

    afferent = np.exp((-(np.abs(sr-r))**2-(np.abs(sz-z))**2)*oneOverTwoSigmaSquared)
    afferent /= afferent.sum()

    # Integrate recurrent dynamics

    z = step(afferent*(sz-z),z)
    r = step(afferent*(sr-r),r)

    if not(t%100): print ('time: '+str(t))

# Compute orientation preference and selectivity

Z = np.arctan2(z.imag,z.real)
Z = (Z+np.where(Z>0.,Z,np.pi))/np.pi
sel = np.abs(z)
sel = sel/np.max(sel)

# Power spectrum using discrete 2D fourier transform

FFT = np.abs(np.fft.fftshift(np.fft.fft2(z)))**2
x,y = np.meshgrid(np.arange(N),np.arange(N))

# Histogram of distances

dist = np.sqrt(pow(x-0.5*(N-1),2.0)+pow(y-0.5*(N-1),2.0))
H = np.histogram(dist.flatten(),weights=FFT.flatten(),bins=50,range=(0,0.5*N))

# Lambda is metric of periodicity of OR map (argmax of histogram)

Lambda = H[1][np.argmax(H[0])+1]

# Count the number of pinwheels above some threshold

dMap = laplace(np.abs(z))
pinsY,pinsX = np.where(((dMap)/np.max(dMap))>thresh)
nPins = len(pinsX)

# Orientation map images

Map = np.zeros([N,N,3])
Sel = np.zeros([N,N,3])
MapSel = np.zeros([N,N,3])
for i in range(N):
    for j in range(N):
        Map[i,j,:] = HSVtoRGB(Z[i,j],1.,1.)
        Sel[i,j,:] = sel[i,j]
        MapSel[i,j,:] = HSVtoRGB(Z[i,j],1.,sel[i,j])

'''
    PLOTTING
'''


F = pl.figure(0,figsize=(12,12))

# Orientation map

f = F.add_subplot(331)
f.imshow(Map,interpolation='none')
f.set_xticks([]), f.set_yticks([])
f.set_title('OR pref.')

# Selectivity

f = F.add_subplot(332)
f.imshow(Sel,interpolation='none')
f.set_xticks([]), f.set_yticks([])
f.set_title('OR sel.')

# Selectivity

f = F.add_subplot(333)
f.imshow(MapSel,interpolation='none')
f.set_xticks([]), f.set_yticks([])
f.set_title('OR pref*sel')

# 2D fourier transform of OR ma

f = F.add_subplot(334)
f.imshow(FFT,interpolation='none',cmap='gray')
f.set_xticks([]), f.set_yticks([])
f.set_title('FFT')

# Histogram (distance vs power in FFT)

f = F.add_subplot(335)
f.plot(H[1][1:],H[0])
f.plot([Lambda,Lambda],[0,np.max(H[0])],'--')
f.set_xticks([]), f.set_yticks([])
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xlabel('distance')
f.set_ylabel('power in FFT')
f.set_title('lambda = '+str(Lambda))


# Laplacian of OR map

f = F.add_subplot(336)
f.imshow(laplace(np.abs(z)),interpolation='none')
f.plot(pinsX,pinsY,'o',markersize=4,markerfacecolor='none')
f.axis(np.array([0,N,0,N])-0.5)
f.set_xticks([]), f.set_yticks([])
f.set_title('Pinwheels = '+str(nPins))

# Fishnet plot (retinotopy)

f = F.add_subplot(337)
f.plot(np.hstack([r.real,r.real.T]),np.hstack([r.imag,r.imag.T]),'.-',color=(0,0,0))
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xticks([]), f.set_yticks([])
f.set_xlabel('x'), f.set_ylabel('y')
f.set_title('retinotopy')

print ('Pinwheel Density = ' + str(1.*nPins/Lambda))

F.tight_layout()

F.savefig('Figure.pdf',dpi=600)
