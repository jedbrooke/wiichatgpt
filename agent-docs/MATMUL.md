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

A: 4,4 RGBA texture
tex0: [ AA_0 AA_1 AA_2 AA_3 | AC_0 AC_1 AC_2 AC_3 | AB_0 AB_1 AB_2 AB_3 | AD_0 AD_1 AD_2 AD_3 
        AA_4 AA_5 AA_6 AA_7 | AC_4 AC_5 AC_6 AC_7 | AB_4 AB_5 AB_6 AB_7 | AD_4 AD_5 AD_6 AD_7
        AA_8 AA_9 AA_A AA_B | AC_8 AC_9 AC_A AC_B | AB_8 AB_9 AB_A AB_B | AD_8 AD_9 AD_A AD_B
        AA_C AA_D AA_E AA_F | AC_C AC_D AC_E AC_F | AB_C AB_D AB_E AB_F | AD_C AD_D AD_E AD_F ]

B: 2,2 RGBA texture
tex1: [BA_0 BA_1 BA_2 BA_3]




write it all out 
AA_0 * BA_0 AA_1 * BA_4 AA_2 * BA_8 AA_3 * BA_C | AC_0 * BA_0 AC_1 * BA_4 AC_2 * BA_8 AC_3 * BA_C | AB_0 * BC_0 AB_1 * BC_4 AB_2 * BC_8 AB_3 * BC_C | AD_0 * BC_0 AD_1 * BC_4 AD_2 * BC_8 AD_3 * BC_C
AA_0 * BA_1 AA_1 * BA_5 AA_2 * BA_9 AA_3 * BA_D | AC_0 * BA_1 AC_1 * BA_5 AC_2 * BA_9 AC_3 * BA_D | AB_0 * BC_1 AB_1 * BC_5 AB_2 * BC_9 AB_3 * BC_D | AD_0 * BC_1 AD_1 * BC_5 AD_2 * BC_9 AD_3 * BC_D
AA_0 * BA_2 AA_1 * BA_6 AA_2 * BA_A AA_3 * BA_E | AC_0 * BA_2 AC_1 * BA_6 AC_2 * BA_A AC_3 * BA_E | AB_0 * BC_2 AB_1 * BC_6 AB_2 * BC_A AB_3 * BC_E | AD_0 * BC_2 AD_1 * BC_6 AD_2 * BC_A AD_3 * BC_E
AA_0 * BA_3 AA_1 * BA_7 AA_2 * BA_B AA_3 * BA_F | AC_0 * BA_3 AC_1 * BA_7 AC_2 * BA_B AC_3 * BA_F | AB_0 * BC_3 AB_1 * BC_7 AB_2 * BC_B AB_3 * BC_F | AD_0 * BC_3 AD_1 * BC_7 AD_2 * BC_B AD_3 * BC_F
AA_0 * BB_0 AA_1 * BB_4 AA_2 * BB_8 AA_3 * BB_C | AC_0 * BB_0 AC_1 * BB_4 AC_2 * BB_8 AC_3 * BB_C | AB_0 * BD_0 AB_1 * BD_4 AB_2 * BD_8 AB_3 * BD_C | AD_0 * BD_0 AD_1 * BD_4 AD_2 * BD_8 AD_3 * BD_C
AA_0 * BB_1 AA_1 * BB_5 AA_2 * BB_9 AA_3 * BB_D | AC_0 * BB_1 AC_1 * BB_5 AC_2 * BB_9 AC_3 * BB_D | AB_0 * BD_1 AB_1 * BD_5 AB_2 * BD_9 AB_3 * BD_D | AD_0 * BD_1 AD_1 * BD_5 AD_2 * BD_9 AD_3 * BD_D
AA_0 * BB_2 AA_1 * BB_6 AA_2 * BB_A AA_3 * BB_E | AC_0 * BB_2 AC_1 * BB_6 AC_2 * BB_A AC_3 * BB_E | AB_0 * BD_2 AB_1 * BD_6 AB_2 * BD_A AB_3 * BD_E | AD_0 * BD_2 AD_1 * BD_6 AD_2 * BD_A AD_3 * BD_E
AA_0 * BB_3 AA_1 * BB_7 AA_2 * BB_B AA_3 * BB_F | AC_0 * BB_3 AC_1 * BB_7 AC_2 * BB_B AC_3 * BB_F | AB_0 * BD_3 AB_1 * BD_7 AB_2 * BD_B AB_3 * BD_F | AD_0 * BD_3 AD_1 * BD_7 AD_2 * BD_B AD_3 * BD_F

AA_4 * BA_0 AA_5 * BA_4 AA_6 * BA_8 AA_7 * BA_C | AC_4 * BA_0 AC_5 * BA_4 AC_6 * BA_8 AC_7 * BA_C | AB_4 * BC_0 AB_5 * BC_4 AB_6 * BC_8 AB_7 * BC_C | AD_4 * BC_0 AD_5 * BC_4 AD_6 * BC_8 AD_7 * BC_C
AA_4 * BA_1 AA_5 * BA_5 AA_6 * BA_9 AA_7 * BA_D | AC_4 * BA_1 AC_5 * BA_5 AC_6 * BA_9 AC_7 * BA_D | AB_4 * BC_1 AB_5 * BC_5 AB_6 * BC_9 AB_7 * BC_D | AD_4 * BC_1 AD_5 * BC_5 AD_6 * BC_9 AD_7 * BC_D
AA_4 * BA_2 AA_5 * BA_6 AA_6 * BA_A AA_7 * BA_E | AC_4 * BA_2 AC_5 * BA_6 AC_6 * BA_A AC_7 * BA_E | AB_4 * BC_2 AB_5 * BC_6 AB_6 * BC_A AB_7 * BC_E | AD_4 * BC_2 AD_5 * BC_6 AD_6 * BC_A AD_7 * BC_E
AA_4 * BA_3 AA_5 * BA_7 AA_6 * BA_B AA_7 * BA_F | AC_4 * BA_3 AC_5 * BA_7 AC_6 * BA_B AC_7 * BA_F | AB_4 * BC_3 AB_5 * BC_7 AB_6 * BC_B AB_7 * BC_F | AD_4 * BC_3 AD_5 * BC_7 AD_6 * BC_B AD_7 * BC_F
AA_4 * BB_0 AA_5 * BB_4 AA_6 * BB_8 AA_7 * BB_C | AC_4 * BB_0 AC_5 * BB_4 AC_6 * BB_8 AC_7 * BB_C | AB_4 * BD_0 AB_5 * BD_4 AB_6 * BD_8 AB_7 * BD_C | AD_4 * BD_0 AD_5 * BD_4 AD_6 * BD_8 AD_7 * BD_C
AA_4 * BB_1 AA_5 * BB_5 AA_6 * BB_9 AA_7 * BB_D | AC_4 * BB_1 AC_5 * BB_5 AC_6 * BB_9 AC_7 * BB_D | AB_4 * BD_1 AB_5 * BD_5 AB_6 * BD_9 AB_7 * BD_D | AD_4 * BD_1 AD_5 * BD_5 AD_6 * BD_9 AD_7 * BD_D
AA_4 * BB_2 AA_5 * BB_6 AA_6 * BB_A AA_7 * BB_E | AC_4 * BB_2 AC_5 * BB_6 AC_6 * BB_A AC_7 * BB_E | AB_4 * BD_2 AB_5 * BD_6 AB_6 * BD_A AB_7 * BD_E | AD_4 * BD_2 AD_5 * BD_6 AD_6 * BD_A AD_7 * BD_E
AA_4 * BB_3 AA_5 * BB_7 AA_6 * BB_B AA_7 * BB_F | AC_4 * BB_3 AC_5 * BB_7 AC_6 * BB_B AC_7 * BB_F | AB_4 * BD_3 AB_5 * BD_7 AB_6 * BD_B AB_7 * BD_F | AD_4 * BD_3 AD_5 * BD_7 AD_6 * BD_B AD_7 * BD_F

AA_8 * BA_0 AA_9 * BA_4 AA_A * BA_8 AA_B * BA_C | AC_8 * BA_0 AC_9 * BA_4 AC_A * BA_8 AC_B * BA_C | AB_8 * BC_0 AB_9 * BC_4 AB_A * BC_8 AB_B * BC_C | AD_8 * BC_0 AD_9 * BC_4 AD_A * BC_8 AD_B * BC_C
AA_8 * BA_1 AA_9 * BA_5 AA_A * BA_9 AA_B * BA_D | AC_8 * BA_1 AC_9 * BA_5 AC_A * BA_9 AC_B * BA_D | AB_8 * BC_1 AB_9 * BC_5 AB_A * BC_9 AB_B * BC_D | AD_8 * BC_1 AD_9 * BC_5 AD_A * BC_9 AD_B * BC_D
AA_8 * BA_2 AA_9 * BA_6 AA_A * BA_A AA_B * BA_E | AC_8 * BA_2 AC_9 * BA_6 AC_A * BA_A AC_B * BA_E | AB_8 * BC_2 AB_9 * BC_6 AB_A * BC_A AB_B * BC_E | AD_8 * BC_2 AD_9 * BC_6 AD_A * BC_A AD_B * BC_E
AA_8 * BA_3 AA_9 * BA_7 AA_A * BA_B AA_B * BA_F | AC_8 * BA_3 AC_9 * BA_7 AC_A * BA_B AC_B * BA_F | AB_8 * BC_3 AB_9 * BC_7 AB_A * BC_B AB_B * BC_F | AD_8 * BC_3 AD_9 * BC_7 AD_A * BC_B AD_B * BC_F
AA_8 * BB_0 AA_9 * BB_4 AA_A * BB_8 AA_B * BB_C | AC_8 * BB_0 AC_9 * BB_4 AC_A * BB_8 AC_B * BB_C | AB_8 * BD_0 AB_9 * BD_4 AB_A * BD_8 AB_B * BD_C | AD_8 * BD_0 AD_9 * BD_4 AD_A * BD_8 AD_B * BD_C
AA_8 * BB_1 AA_9 * BB_5 AA_A * BB_9 AA_B * BB_D | AC_8 * BB_1 AC_9 * BB_5 AC_A * BB_9 AC_B * BB_D | AB_8 * BD_1 AB_9 * BD_5 AB_A * BD_9 AB_B * BD_D | AD_8 * BD_1 AD_9 * BD_5 AD_A * BD_9 AD_B * BD_D
AA_8 * BB_2 AA_9 * BB_6 AA_A * BB_A AA_B * BB_E | AC_8 * BB_2 AC_9 * BB_6 AC_A * BB_A AC_B * BB_E | AB_8 * BD_2 AB_9 * BD_6 AB_A * BD_A AB_B * BD_E | AD_8 * BD_2 AD_9 * BD_6 AD_A * BD_A AD_B * BD_E
AA_8 * BB_3 AA_9 * BB_7 AA_A * BB_B AA_B * BB_F | AC_8 * BB_3 AC_9 * BB_7 AC_A * BB_B AC_B * BB_F | AB_8 * BD_3 AB_9 * BD_7 AB_A * BD_B AB_B * BD_F | AD_8 * BD_3 AD_9 * BD_7 AD_A * BD_B AD_B * BD_F

AA_C * BA_0 AA_D * BA_4 AA_E * BA_8 AA_F * BA_C | AC_C * BA_0 AC_D * BA_4 AC_E * BA_8 AC_F * BA_C | AB_C * BC_0 AB_D * BC_4 AB_E * BC_8 AB_F * BC_C | AD_C * BC_0 AD_D * BC_4 AD_E * BC_8 AD_F * BC_C
AA_C * BA_1 AA_D * BA_5 AA_E * BA_9 AA_F * BA_D | AC_C * BA_1 AC_D * BA_5 AC_E * BA_9 AC_F * BA_D | AB_C * BC_1 AB_D * BC_5 AB_E * BC_9 AB_F * BC_D | AD_C * BC_1 AD_D * BC_5 AD_E * BC_9 AD_F * BC_D
AA_C * BA_2 AA_D * BA_6 AA_E * BA_A AA_F * BA_E | AC_C * BA_2 AC_D * BA_6 AC_E * BA_A AC_F * BA_E | AB_C * BC_2 AB_D * BC_6 AB_E * BC_A AB_F * BC_E | AD_C * BC_2 AD_D * BC_6 AD_E * BC_A AD_F * BC_E
AA_C * BA_3 AA_D * BA_7 AA_E * BA_B AA_F * BA_F | AC_C * BA_3 AC_D * BA_7 AC_E * BA_B AC_F * BA_F | AB_C * BC_3 AB_D * BC_7 AB_E * BC_B AB_F * BC_F | AD_C * BC_3 AD_D * BC_7 AD_E * BC_B AD_F * BC_F
AA_C * BB_0 AA_D * BB_4 AA_E * BB_8 AA_F * BB_C | AC_C * BB_0 AC_D * BB_4 AC_E * BB_8 AC_F * BB_C | AB_C * BD_0 AB_D * BD_4 AB_E * BD_8 AB_F * BD_C | AD_C * BD_0 AD_D * BD_4 AD_E * BD_8 AD_F * BD_C
AA_C * BB_1 AA_D * BB_5 AA_E * BB_9 AA_F * BB_D | AC_C * BB_1 AC_D * BB_5 AC_E * BB_9 AC_F * BB_D | AB_C * BD_1 AB_D * BD_5 AB_E * BD_9 AB_F * BD_D | AD_C * BD_1 AD_D * BD_5 AD_E * BD_9 AD_F * BD_D
AA_C * BB_2 AA_D * BB_6 AA_E * BB_A AA_F * BB_E | AC_C * BB_2 AC_D * BB_6 AC_E * BB_A AC_F * BB_E | AB_C * BD_2 AB_D * BD_6 AB_E * BD_A AB_F * BD_E | AD_C * BD_2 AD_D * BD_6 AD_E * BD_A AD_F * BD_E
AA_C * BB_3 AA_D * BB_7 AA_E * BB_B AA_F * BB_F | AC_C * BB_3 AC_D * BB_7 AC_E * BB_B AC_F * BB_F | AB_C * BD_3 AB_D * BD_7 AB_E * BD_B AB_F * BD_F | AD_C * BD_3 AD_D * BD_7 AD_E * BD_B AD_F * BD_F


