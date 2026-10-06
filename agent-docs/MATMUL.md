let's look at a 4x4 matmul

```
[A0 A1 A2 A3]   [B0 B1 B2 B3]   [(A0*B0 + A1*B4 + A2*B8 + A3*BC)  (A0*B1 + A1*B5 + A2*B9 + A3*BD)  (A0*B2 + A1*B6 + A2*BA + A3*BE)  (A0*B3 + A1*B7 + A2*BB + A3*BF)]
[A4 A5 A6 A7] x [B4 B5 B6 B7] = [(A4*B0 + A5*B4 + A6*B8 + A7*BC)  (A4*B1 + A5*B5 + A6*B9 + A7*BD)  (A4*B2 + A5*B6 + A6*BA + A7*BE)  (A4*B3 + A5*B7 + A6*BB + A7*BF)]
[A8 A9 AA AB]   [B8 B9 BA BB]   [(A8*B0 + A9*B4 + AA*B8 + AB*BC)  (A8*B1 + A9*B5 + AA*B9 + AB*BD)  (A8*B2 + A9*B6 + AA*BA + AB*BE)  (A8*B3 + A9*B7 + AA*BB + AB*BF)]
[AC AD AE AF]   [BC BD BE BF]   [(AC*B0 + AD*B4 + AE*B8 + AF*BC)  (AC*B1 + AD*B5 + AE*B9 + AF*BD)  (AC*B2 + AD*B6 + AE*BA + AF*BE)  (AC*B3 + AD*B7 + AE*BB + AF*BF)]

tex0: [A0 A0 A0 A0 A4 A4 A4 A4 A8 A8 A8 A8 AC AC AC AC]
tex1: [B0 B1 B2 B3 B0 B1 B2 B3 B0 B1 B2 B3 B0 B1 B2 B3]

tex2: [A1 A1 A1 A1 A5 A5 A5 A5 A9 A9 A9 A9 AD AD AD AD]
tex3: [B4 B5 B6 B7 B4 B5 B6 B7 B4 B5 B6 B7 B4 B5 B6 B7]

tex2: [A2 A2 A2 A2 A6 A6 A6 A6 AA AA AA AA AE AE AE AE]
tex3: [B8 B9 BA BB B8 B9 BA BB B8 B9 BA BB B8 B9 BA BB]

tex2: [A3 A3 A3 A3 A7 A7 A7 A7 AB AB AB AB AF AF AF AF]
tex3: [BC BD BE BF BC BD BE BF BC BD BE BF BC BD BE BF]

final EFB: [C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 CA CB CC CD CE CF]

to keep the later accumulates simple, let's cap it to 1024 x 1024 (even thogh our hardware can go up to 1280 x 1026), we like powers of 2 :).

Imagine a larger matrix now, A[i][j] and a matric B[j][k].  TODO