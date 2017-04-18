#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as pl

from numpy import random as rn

N = 4096
iscale = 8

nx = N
ny = N * iscale
dx = rn.randn(ny)
f1 = np.sum(dx**2)/dx.size
DX = np.fft.rfft(dx)
j = np.linspace(0, ny/2, ny/2+1)
CFX = np.exp(-0.5 * (j/100.)**2)
MDX = CFX*DX
mdx = np.sqrt(N)*np.fft.irfft(MDX)

x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)

sheet = np.linspace(0,nx-1,ny) + mdx
np.savetxt('sheet.dat', sheet, header='Corrugated current sheet surface', comments='#')
#np.savetxt('sheet.dat', sheet)
rho = np.zeros((nx,ny))
sigma = 1.0
rsum = np.zeros((ny))

for j in range(0,ny-1):
    i0 = max(0.0, sheet[j]-4*sigma)
    i1 = min(nx-1, sheet[j]+4*sigma)
    for i in range(int(round(i0)),int(round(i1))):
        rho[i,j] = np.exp(-0.5*((i-sheet[j])/sigma)**2)/sigma
        rsum[j] += rho[i,j]
    rho[:,j] *= np.sqrt(2*np.pi)/rsum[j]

rho1 = np.sum(rho,axis=1)
rho1 *= nx/np.sum(rho,axis=None)
np.savetxt('rho.dat', rho1, header='Projected density', comments='#')

rho1 *= 1000 * 2
dz = np.gradient(rho1)
dt = np.linspace(0, nx-1, nx) + dz

xi = np.linspace(0,ny-1,ny*1000)
for idx in range(0,xi.size-1):
    i = xi[idx]
    ii = xi[idx]/iscale + 1
    if(ii+1>nx):
        continue
    yy = xi[idx]/iscale+1
    jm = yy
    jp = jm+1
    w = jp-yy
    dz1=w*dz[np.int(jm)]+(1-w)*dz[np.int(jp)]
    jj=np.round(dz1+yy)
    if(jj>nx or jj<1):
        continue
    ak2=dz[np.int(jp)]-dz[np.int(jm)]

flux = np.zeros((ny))
cnt = np.zeros((ny))
o = np.ones(rho.shape, dtype="float")
for i in range(0,ny-1,iscale):
    for i1 in range(i,i+iscale-1):
        if rho[nx/2,i1] != 0.0:
            cnt[i] += 1.0
        flux[i] = np.sum(rho[nx/2,i:i+iscale-1])/(cnt[i] + 1e-13)


#f = pl.figure()
#pl.subplots_adjust(hspace=0.001)


xmin = 0.45
xmax = 0.55

ax1=pl.subplot(311)
ax1.plot(y,sheet, 'k')
#ax1.set_ylabel('Position')
pl.xlim(xmin, xmax)
pl.ylim(1800,2200)
pl.ylim(sheet[np.int(xmin*ny)], sheet[np.int(xmax*ny)])

ax2=pl.subplot(312)
ax2.plot(x,rho1,'k')
ax2.set_ylabel('Projected density')
pl.xlim(xmin, xmax)
#pl.ylim(0,2000)
#pl.ylim(rho1[np.int(xmin*nx)], rho1[np.int(xmax*nx)])

ax3=pl.subplot(313)
ax3.plot(x,dt,'k')
ax3.set_xlabel('Position')
ax3.set_ylabel('Deflection')
pl.xlim(xmin, xmax)
pl.ylim(dt[np.int(xmin*nx)], dt[np.int(xmax*nx)])
pl.ylim(1800,2200)

pl.show()

