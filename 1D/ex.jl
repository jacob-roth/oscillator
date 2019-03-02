Mg = 10.0
Dg = 0.05
Dl = 0.05
Dv = 0.05
B01 = 10.0
B00 = -10.0
Q10 = -0.5
P0 = -4.0

function integrate(ww, aa, vv; dt=0.01, n=1000)
    h = zeros(3, n)
    w = copy(ww)
    a = copy(aa)
    v = copy(vv)
    h[:, 1] = [w; a; v]
    for i = 2:n
        w += ((1.0/Mg) * (Dg*w + P0 - B01*v*sin(a))) * dt
        a += (-(1.0/Dl) * (P0 - B01*v*sin(a)) - w) * dt
        a = mod(a, 2*pi)
        v += ((1.0/Dv) * ((1.0/v) * (-Q10 - B01*v*cos(a) - B00*v^2))) * dt
        h[:, i] = [w; a; v]
        println(h[:, i])
    end
    return h
end

w = 0.0
a = pi/3
v = 0.4
h = integrate(w,a,v; dt=0.001, n=100)
