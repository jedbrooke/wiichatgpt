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

8,4 RGBA4 texture 
tex0: [ AA_0 AA_1 AA_2 AA_3 | AA_4 AA_5 AA_6 AA_7 | AB_0 AB_1 AB_2 AB_3 | AB_4 AB_5 AB_6 AB_7 | AA_0 AA_1 AA_2 AA_3 | AA_4 AA_5 AA_6 AA_7 | AB_0 AB_1 AB_2 AB_3 | AB_4 AB_5 AB_6 AB_7
        AA_8 AA_9 AA_A AA_B | AA_C AA_D AA_E AA_F | AB_8 AB_9 AB_A AB_B | AB_C AB_D AB_E AB_F | AA_8 AA_9 AA_A AA_B | AA_C AA_D AA_E AA_F | AB_8 AB_9 AB_A AB_B | AB_C AB_D AB_E AB_F 
        AC_0 AC_1 AC_2 AC_3 | AC_4 AC_5 AC_6 AC_7 | AD_0 AD_1 AD_2 AD_3 | AD_4 AD_5 AD_6 AD_7 | AC_0 AC_1 AC_2 AC_3 | AC_4 AC_5 AC_6 AC_7 | AD_0 AD_1 AD_2 AD_3 | AD_4 AD_5 AD_6 AD_7
        AC_8 AC_9 AC_A AC_B | AC_C AC_D AC_E AC_F | AD_8 AD_9 AD_A AD_B | AD_C AD_D AD_E AD_F | AC_8 AC_9 AC_A AC_B | AC_C AC_D AC_E AC_F | AD_8 AD_9 AD_A AD_B | AD_C AD_D AD_E AD_F ]
4,2 RGBA texture
tex1: [ BA_0 BA_1 BA_2 BA_3 | BC_0 BC_1 BC_2 BC_3 | BB_0 BB_1 BB_2 BB_3 | BD_0 BD_1 BD_2 BD_3
        BA_0 BA_1 BA_2 BA_3 | BC_0 BC_1 BC_2 BC_3 | BB_0 BB_1 BB_2 BB_3 | BD_0 BD_1 BD_2 BD_3 ]
tex2: [ BA_4 BA_5 BA_6 BA_7 | BC_4 BC_5 BC_6 BC_7 | BB_4 BB_5 BB_6 BB_7 | BD_4 BD_5 BD_6 BD_7
        BA_4 BA_5 BA_6 BA_7 | BC_4 BC_5 BC_6 BC_7 | BB_4 BB_5 BB_6 BB_7 | BD_4 BD_5 BD_6 BD_7 ] 
tex3: [ BA_8 BA_9 BA_A BA_B | BC_8 BC_9 BC_A BC_B | BB_8 BB_9 BB_A BB_B | BD_8 BD_9 BD_A BD_B
        BA_8 BA_9 BA_A BA_B | BC_8 BC_9 BC_A BC_B | BB_8 BB_9 BB_A BB_B | BD_8 BD_9 BD_A BD_B ] 
tex4: [ BA_C BA_D BA_E BA_F | BC_C BC_D BC_E BC_F | BB_C BB_D BB_E BB_F | BD_C BD_D BD_E BD_F
        BA_C BA_D BA_E BA_F | BC_C BC_D BC_E BC_F | BB_C BB_D BB_E BB_F | BD_C BD_D BD_E BD_F ]

8,4
EFB = [ (AA x BA) | (AB x BC) | (AA x BB) | (AB x BD) ]
      [ (AC x BA) | (AD x BC) | (AD x BB) | (AD x BD) ]

in theory our actual mem requirement is only 4,4 for A and 4x 2,2 for B = 32, compared to 64 now (A and B are needed twice). but we'd still need 8,4 output. 
stretch tex0 [x, 2 * y], stretch tex1-4 [2 * x, 2 * y].
Stretching Btex on y is easy, since that's a straigh dupliate, we already do it for x. Stretching A tex presents an issue, since we need to repeat rather than just a simple stretch. Or if we stretch A and B together then it just does the same computation again, not actualy crossing A and B. We could use a indirect texture lookup, or maybe mess around with INT8 vs RGBA8 texture type.
This might just be premature optimization though, since B already gives us half of the improvement. We might not even be memory bandwidth bound

4,4 RGBA4 texture 
tex0: [ AA_0 AA_1 AA_2 AA_3 | AA_4 AA_5 AA_6 AA_7 | AB_0 AB_1 AB_2 AB_3 | AB_4 AB_5 AB_6 AB_7 |
        AA_8 AA_9 AA_A AA_B | AA_C AA_D AA_E AA_F | AB_8 AB_9 AB_A AB_B | AB_C AB_D AB_E AB_F | 
        AC_0 AC_1 AC_2 AC_3 | AC_4 AC_5 AC_6 AC_7 | AD_0 AD_1 AD_2 AD_3 | AD_4 AD_5 AD_6 AD_7 |
        AC_8 AC_9 AC_A AC_B | AC_C AC_D AC_E AC_F | AD_8 AD_9 AD_A AD_B | AD_C AD_D AD_E AD_F ]
8,1 GX_TF_IA8 texture
tex1: [ BA_0 BA_1 | BA_2 `BA_3 | BC_0 BC_1 | BC_2 BC_3 | BB_0 BB_1 | BB_2 BB_3 | BD_0 BD_1 | BD_2 BD_3 ]
tex2: [ BA_4 BA_5 | BA_6 BA_7 | BC_4 BC_5 | BC_6 BC_7 | BB_4 BB_5 | BB_6 BB_7 | BD_4 BD_5 | BD_6 BD_7 ]
tex3: [ BA_8 BA_9 | BA_A BA_B | BC_8 BC_9 | BC_A BC_B | BB_8 BB_9 | BB_A BB_B | BD_8 BD_9 | BD_A BD_B ]
tex4: [ BA_C BA_D | BA_E BA_F | BC_C BC_D | BC_E BC_F | BB_C BB_D | BB_E BB_F | BD_C BD_D | BD_E BD_F ]


Stage 0: TEV_PREV = swap_tables tex0[x/2,y][R] 
Stage 1: TEV_PREV = TEV_PREV * tex1[x/2,y/4]  # [ (AA_0 * BA_0) .. | (AA_4 * BA_0) .. | (AB_0 * BC_0) .. | .. |
                                                  (AA_8 * BA_0) .. | (AA_C * BA_0) .. | (AB_8 * BC_0) .. | .. |
                                                  (AC_0 * BA_0) .. | (AC_4 * BA_0) .. | (AB_0 * BC_0) .. | .. |
                                                  (AC_8 * BA_0) .. | (AC_C * BA_0) .. | (AB_8 * BC_0) .. | .. | ]
Stage 2: TEV_REG1 = swap_tables tex0[x/2,y][G]
Stage 3: TEV_PREV += TEV_REG1 * tex2[x/2,y/4] 
Stage 4: TEV_REG1 = swap_tables tex0[x/2,y][B]
Stage 5: TEV_PREV += TEV_REG1 * tex3[x/2,y/4] 
Stage 6: TEV_REG1 = swap_tables tex0[x/2,y][A]
Stage 7: TEV_PREV += TEV_REG1 * tex4[x/2,y/4] 
Stage 8: TEV_PREV /= 2
Stage 9: TEV_PREV /= 2

8,4
final EFB: [ P0_0 P0_1 P0_2 P0_3 | P0_4 P0_5 P0_6 P0_7 | P1_0 P1_1 P1_2 P1_3 | P1_4 P1_5 P1_6 P1_7 | P2_0 P2_1 P2_2 P2_3 | P2_4 P2_5 P2_6 P2_7 | P3_0 P3_1 P3_2 P3_3 | P3_4 P3_5 P3_6 P3_7 |
             P0_8 P0_9 P0_A P0_B | P0_C P0_D P0_E P0_F | P1_8 P1_9 P1_A P1_B | P1_C P1_D P1_E P1_F | P2_8 P2_9 P2_A P2_B | P2_C P2_D P2_E P2_F | P3_8 P3_9 P3_A P3_B | P3_C P3_D P3_E P3_F |
             P4_0 P4_1 P4_2 P4_3 | P4_4 P4_5 P4_6 P4_7 | P5_0 P5_1 P5_2 P5_3 | P5_4 P5_5 P5_6 P5_7 | P6_0 P6_1 P6_2 P6_3 | P6_4 P6_5 P6_6 P6_7 | P7_0 P7_1 P7_2 P7_3 | P7_4 P7_5 P7_6 P7_7 |
             P4_8 P4_9 P4_A P4_B | P4_C P4_D P4_E P4_F | P5_8 P5_9 P5_A P5_B | P5_C P5_D P5_E P5_F | P6_8 P6_9 P6_A P6_B | P6_C P6_D P6_E P6_F | P7_8 P7_9 P7_A P7_B | P7_C P7_D P7_E P7_F ]