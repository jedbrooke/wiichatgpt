# Overview
GX TEV only gives us a u8 output with s10 intermediate representaion. This works fine for the MACs if we are willing to give up some precision, however for the final accumulate for large dp/matmuls, u8 just really isn't enough

## TYPES
u8 -> standard unsigned 8bit
fint8 -> 8 bit num that's treated as 0.0-1.0 for mul. (eg 255 * 127 = 127) (c * b + 128) >> 8
s10 -> 11 bit signed int format [-1024..1023]

higher bit accumulation can be done using only 8 bit registers as follows

### 7 bit limbs
a = limbs[l]
b = limbs[l]
out = limbs[l]
carry = 0
for i in (l-1)..0:
    t = a[i] + b[i] + carry
    out[i] = t & 0x7f
    carry = t >> 7


for i in (l-1)..0:
    roughly in TEV Stages
    draw call 0
    load 0x7f into the efb

    draw call 1
    tex0: a[i], tex1: b[i], tex2: carry
    Stage0: load tex0 into TEV_PREV
    Stage1: TEV_PREV = TEV_PREV + tex1
    Stage2: TEV_PREV = TEV_PREV + carry
    write TEV_PREV to efb, AND with existing

    copyTex efb to a texture (out[i])

    draw call 2
    tex0: a[i], tex1: b[i], tex2: carry
    Stage0: load tex0 into TEV_PREV
    Stage1: TEV_PREV = TEV_PREV + tex1
    Stage2: TEV_PREV = (TEV_PREV + carry)/2
    stage 3-8: TEV_PREV = TEV_PREV/2
    write TEV_PREV to efb

    copyTex efb to a texture (carry)


However, our data coming in is 8 bits, so we'd need to first extract it to 7 bit limbs 

roughly in TEV Stages
draw call 0
load 0x7f into the efb

draw call 1
tex0: a
Stage0: load tex0 into TEV_PREV
write TEV_PREV to efb, AND with existing

copyTex efb to texture (a[0])

draw call 2
tex0: a
Stage0: load tex0 into TEV_PREV and /2
Stage 1-6: TEV_PREV = TEV_PREV /2
write TEV_PREV to efb

copyTex efb to texture (a[1])

this adds only 3 draw calls of setup, if we count both operands then it's 6 setup for 21 total.

For 32 bits, we have 3 setup + 5 loops * 3 draw calls = 18, if we're ok with 28 bits instead of 32, then we'd only need 4 loops which gives 13 total (16 if we count both inputs for setup). 


### Graveyard
we considered carry extraction using carry extraction

out = limbs[l]
a = limbs[l]
b = limbs[l]
c = 0
for i in (l-1)..0:
    out[i] = a[i] + b[i] + c[i]
    c = ((a[i] & b[i]) | ((a[i] | b[i]) & ~out[i])) >> 7

however, 8 bit limbs will suffer from precision issues when using lerp for addition, and it will cost too many draw calls to do all the bit arithmetic.

### Need empirical data
for the `out[i] = t & 0x7f` stage, we use an extra draw call to load 0x7f into the EFB, this still let's us to big parallel on the gpu, but we'd have to see if that's actually faster than just doing it on the cpu after readback.