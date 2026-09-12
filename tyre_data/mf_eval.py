import re, math, sys

def load(fn):
    p={}
    for line in open(fn, errors='replace'):
        line=line.split('$')[0]
        m=re.match(r"\s*([A-Za-z_0-9]+)\s*=\s*(-?[\d.eE+-]+)\s*$", line)
        if m:
            try: p[m.group(1).upper()]=float(m.group(2))
            except: pass
    return p

def g(p,k,d=0.0): return p.get(k,d)

def fy0(p, Fz, alpha, gamma=0.0, mf6=False):
    Fz0p = g(p,'LFZO',1.0)*g(p,'FNOMIN',4000.0)
    dfz  = (Fz-Fz0p)/Fz0p
    dpi  = 0.0
    gy   = gamma*g(p,'LGAY',1.0)
    Cy   = g(p,'PCY1',1.3)*g(p,'LCY',1.0)
    muy  = (g(p,'PDY1')+g(p,'PDY2')*dfz)*(1+g(p,'PPY3')*dpi+g(p,'PPY4')*dpi**2) \
           *(1-g(p,'PDY3')*gy**2)*g(p,'LMUY',1.0)
    Dy   = muy*Fz
    if mf6:
        denom=(g(p,'PKY2')+g(p,'PKY5')*gy**2)*Fz0p*(1+g(p,'PPY2')*dpi)
        Kya = g(p,'PKY1')*Fz0p*(1+g(p,'PPY1')*dpi)*(1-g(p,'PKY3')*abs(gy)) \
              *math.sin(g(p,'PKY4',2.0)*math.atan(Fz/denom))*g(p,'LKY',1.0)
    else:
        Kya = g(p,'PKY1')*Fz0p*math.sin(2*math.atan(Fz/(g(p,'PKY2')*Fz0p))) \
              *(1-g(p,'PKY3')*abs(gy))*g(p,'LFZO',1.0)*g(p,'LKY',1.0)
    By   = Kya/(Cy*Dy) if abs(Cy*Dy)>1e-9 else 0.0
    SHy  = (g(p,'PHY1')+g(p,'PHY2')*dfz)*g(p,'LHY',1.0)+g(p,'PHY3')*gy
    SVy  = Fz*((g(p,'PVY1')+g(p,'PVY2')*dfz)*g(p,'LVY',1.0)
               +(g(p,'PVY3')+g(p,'PVY4')*dfz)*gy)*g(p,'LMUY',1.0)
    ay   = alpha+SHy
    s    = 1.0 if ay>=0 else -1.0
    Ey   = (g(p,'PEY1')+g(p,'PEY2')*dfz)*(1+g(p,'PEY5')*gy**2-(g(p,'PEY3')+g(p,'PEY4')*gy)*s)*g(p,'LEY',1.0)
    Ey   = min(Ey,1.0)
    x    = By*ay
    return Dy*math.sin(Cy*math.atan(x-Ey*(x-math.atan(x))))+SVy, Kya, muy

def peak(p,Fz,mf6):
    best=0.0; ba=0.0
    a=-0.001
    while a>-0.45:
        f,_,_=fy0(p,Fz,a,0.0,mf6)
        if abs(f)>best: best=abs(f); ba=a
        a-=0.0005
    a=0.001
    while a<0.45:
        f,_,_=fy0(p,Fz,a,0.0,mf6)
        if abs(f)>best: best=abs(f); ba=a
        a+=0.0005
    return best,ba

files=[('tir/TNO_car205_60R15.tir','TNO/Delft 205/60R15 (MF6.2, dry asphalt)',True),
       ('tir/pac2002_235_60R16.tir','MSC Pac2002 example 235/60R16',False),
       ('tir/mf_185_80R14.tir','Chrono VW microbus 185/80R14',False),
       ('tir/Sedan_Pac02Tire.tir','Chrono Sedan 245/40R18 (LFZO 0.81)',False)]
for fn,name,mf6 in files:
    p=load(fn)
    print('='*78); print(name)
    print(' FNOMIN=%.0f N  LFZO=%.3g  Fz0p=%.0f N  PCY1=%.4g PDY1=%.4g PDY2=%.4g'%(
        g(p,'FNOMIN'),g(p,'LFZO',1),g(p,'LFZO',1)*g(p,'FNOMIN'),g(p,'PCY1'),g(p,'PDY1'),g(p,'PDY2')))
    print('  Fz[N]   muy(D-term)  peakFy[N]  peak|Fy|/Fz  alpha_pk[deg]  Cf=|Kya|[N/rad]  Cf/Fz[1/rad]')
    rows=[]
    for Fz in [1000,1500,2000,2500,3000,3500,4000,5000,6000,8000]:
        pk,ba=peak(p,Fz,mf6)
        _,Kya,muy=fy0(p,Fz,ba,0.0,mf6)
        rows.append((Fz,muy,pk,pk/Fz,math.degrees(ba),abs(Kya)))
        print('  %5d   %8.4f   %8.1f   %8.4f     %6.2f        %9.0f      %6.2f'%(
            Fz,muy,pk,pk/Fz,math.degrees(ba),abs(Kya),abs(Kya)/Fz))
    d=dict((r[0],r[3]) for r in rows)
    print('  --> mu(2000)=%.4f  mu(4000)=%.4f  DROP 2000->4000 N = %.2f%%'%(
        d[2000],d[4000],100*(d[4000]-d[2000])/d[2000]))
    print('  --> d(mu)/dFz over 2-4kN = %.2f per MN'%(1e6*(d[4000]-d[2000])/2000))
    print('  --> mu(2700)~%.4f (interp)'%( (d[2500]+ (d[3000]-d[2500])*(2700-2500)/500) ))
    n=math.log(d[4000]*4000/(d[2000]*2000))/math.log(2)
    print('  --> power-law exponent Fy ~ Fz^%.3f over 2-4kN'%n)

print()
print('#'*78)
print('MARGINAL mu  d(Fy_peak)/d(Fz)  == effective payoff of 1 N of DOWNFORCE')
print('(a LATERAL aero force pays 1.000 N of cornering force per N, by definition)')
for fn,name,mf6 in files:
    p=load(fn); print('-'*78); print(name)
    print('   Fz[N]    mu=Fy/Fz    mu_marginal=dFy/dFz    lateral advantage = 1/mu_marg')
    for Fz in [2000,2722,3500,4500,5000,5500,6000]:
        h=25.0
        f1,_=peak(p,Fz-h,mf6); f2,_=peak(p,Fz+h,mf6); f0,_=peak(p,Fz,mf6)
        marg=(f2-f1)/(2*h)
        print('   %5d     %.4f          %.4f                 %.3f'%(Fz,f0/Fz,marg,1.0/marg))
