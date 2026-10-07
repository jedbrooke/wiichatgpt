let's look at a 4x4 matmul

```
[A0 A1 A2 A3]   [B0 B1 B2 B3]   [(A0*B0 + A1*B4 + A2*B8 + A3*BC)  (A0*B1 + A1*B5 + A2*B9 + A3*BD)  (A0*B2 + A1*B6 + A2*BA + A3*BE)  (A0*B3 + A1*B7 + A2*BB + A3*BF)]
[A4 A5 A6 A7] x [B4 B5 B6 B7] = [(A4*B0 + A5*B4 + A6*B8 + A7*BC)  (A4*B1 + A5*B5 + A6*B9 + A7*BD)  (A4*B2 + A5*B6 + A6*BA + A7*BE)  (A4*B3 + A5*B7 + A6*BB + A7*BF)]
[A8 A9 AA AB]   [B8 B9 BA BB]   [(A8*B0 + A9*B4 + AA*B8 + AB*BC)  (A8*B1 + A9*B5 + AA*B9 + AB*BD)  (A8*B2 + A9*B6 + AA*BA + AB*BE)  (A8*B3 + A9*B7 + AA*BB + AB*BF)]
[AC AD AE AF]   [BC BD BE BF]   [(AC*B0 + AD*B4 + AE*B8 + AF*BC)  (AC*B1 + AD*B5 + AE*B9 + AF*BD)  (AC*B2 + AD*B6 + AE*BA + AF*BE)  (AC*B3 + AD*B7 + AE*BB + AF*BF)]

2,2 RGBA4 texture 
tex0: [ A0 A1 A2 A3 | A4 A5 A6 A7 | 
        A8 A9 AA AB | AC AD AE AF ]
1,1 RGBA texture
tex1: [B0 B1 B2 B3]
tex2: [B4 B5 B6 B7] 
tex3: [B8 B9 BA BB] 
tex4: [BC BD BE BF]

Stage 0: TEV_PREV = swap_tables tex0[x,y][R] 
Stage 1: TEV_PREV = TEV_PREV * tex1[x/2,y/2]  # [ (A0 * B0) (A0 * B1) (A0 * B2) (A0 * B3) | (A4 * B0) (A4 * B1) (A4 * B2) (A4 * B3) |
                                                  (A8 * B0) (A8 * B1) (A8 * B2) (A8 * B3) | (AC * B0) (AC * B1) (AC * B2) (AC * B3) ]
Stage 2: TEV_REG1 = swap_tables tex0[x,y][G]
Stage 3: TEV_PREV += TEV_REG1 * tex2[x/2,y/2] # [ (A1 * B4) (A1 * B5) (A1 * B6) (A1 * B7) | (A5 * B4) (A5 * B5) (A5 * B6) (A5 * B7) |
                                                  (A9 * B4) (A9 * B5) (A9 * B6) (A9 * B7) | (AD * B4) (AD * B5) (AD * B6) (AD * B7) ]
Stage 4: TEV_REG1 = swap_tables tex0[x,y][B]
Stage 5: TEV_PREV += TEV_REG1 * tex3[x/2,y/2] # [ (A2 * B8) (A2 * B9) (A2 * BA) (A2 * BB) | (A6 * B8) (A6 * B9) (A6 * BA) (A6 * BB) |
                                                  (AA * B8) (AA * B9) (AA * BA) (AA * BB) | (AE * B8) (AE * B9) (AE * BA) (AE * BB) ]
Stage 6: TEV_REG1 = swap_tables tex0[x,y][A]
Stage 7: TEV_PREV += TEV_REG1 * tex4[x/2,y/2] # [ (A3 * BC) (A3 * BD) (A3 * BE) (A3 * BF) | (A7 * BC) (A7 * BD) (A7 * BE) (A7 * BF) |
                                                  (AB * BC) (AB * BD) (AB * BE) (AB * BF) | (AF * BC) (AF * BD) (AF * BE) (AF * BF) ]
Stage 8: TEV_PREV /= 2
Stage 9: TEV_PREV /= 2

final EFB: [ C0 C1 C2 C3 | C4 C5 C6 C7 |
             C8 C9 CA CB | CC CD CE CF ]

to keep the later accumulates simple, tex A can go up to 512 x 512 px (1M values) and B will be 256 x 256 (262k per tex, 1M total for 4 tex). hardware could do a little higher (640x528) but powers of 2 will probably make the accumulator simpler.

Imagine a larger matrix now, A[8][8] and a matrix B[8][8].
where x is matmul and + is elementwise add

[ AA AB ] x [ BA BB ] = [ (AA x BA) + (AB x BC)  (AA x BB) + (AB x BD)]
[ AC AD ]   [ BC BD ]   [ (AC x BA) + (AD x BC)  (AD x BB) + (AD x BD)]