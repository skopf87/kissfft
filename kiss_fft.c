/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include <stdint.h>
#include "_kiss_fft_guts.h"
/* The guts header contains all the multiplication and addition macros that are defined for
 fixed or floating point complex numbers.  It also delares the kf_ internal functions.
 */

static void kf_bfly2(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    kiss_fft_cpx * Fout2;
    kiss_fft_cpx * tw1 = st->twiddles;
    kiss_fft_cpx t;
    Fout2 = Fout + m;
    do{
        C_FIXDIV(*Fout,2); C_FIXDIV(*Fout2,2);

        C_MUL (t,  *Fout2 , *tw1);
        tw1 += fstride;
        C_SUB( *Fout2 ,  *Fout , t );
        C_ADDTO( *Fout ,  t );
        ++Fout2;
        ++Fout;
    }while (--m);
}

static void kf_bfly4(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        const size_t m
        )
{
    kiss_fft_cpx *tw1,*tw2,*tw3;
    kiss_fft_cpx scratch[6];
    size_t k=m;
    const size_t m2=2*m;
    const size_t m3=3*m;


    tw3 = tw2 = tw1 = st->twiddles;

    do {
        C_FIXDIV(*Fout,4); C_FIXDIV(Fout[m],4); C_FIXDIV(Fout[m2],4); C_FIXDIV(Fout[m3],4);

        C_MUL(scratch[0],Fout[m] , *tw1 );
        C_MUL(scratch[1],Fout[m2] , *tw2 );
        C_MUL(scratch[2],Fout[m3] , *tw3 );

        C_SUB( scratch[5] , *Fout, scratch[1] );
        C_ADDTO(*Fout, scratch[1]);
        C_ADD( scratch[3] , scratch[0] , scratch[2] );
        C_SUB( scratch[4] , scratch[0] , scratch[2] );
        C_SUB( Fout[m2], *Fout, scratch[3] );
        tw1 += fstride;
        tw2 += fstride*2;
        tw3 += fstride*3;
        C_ADDTO( *Fout , scratch[3] );

        if(st->inverse) {
            Fout[m].r = scratch[5].r - scratch[4].i;
            Fout[m].i = scratch[5].i + scratch[4].r;
            Fout[m3].r = scratch[5].r + scratch[4].i;
            Fout[m3].i = scratch[5].i - scratch[4].r;
        }else{
            Fout[m].r = scratch[5].r + scratch[4].i;
            Fout[m].i = scratch[5].i - scratch[4].r;
            Fout[m3].r = scratch[5].r - scratch[4].i;
            Fout[m3].i = scratch[5].i + scratch[4].r;
        }
        ++Fout;
    }while(--k);
}

static void kf_bfly3(
         kiss_fft_cpx * Fout,
         const size_t fstride,
         const kiss_fft_cfg st,
         size_t m
         )
{
     size_t k=m;
     const size_t m2 = 2*m;
     kiss_fft_cpx *tw1,*tw2;
     kiss_fft_cpx scratch[5];
     kiss_fft_cpx epi3;
     epi3 = st->twiddles[fstride*m];

     tw1=tw2=st->twiddles;

     do{
         C_FIXDIV(*Fout,3); C_FIXDIV(Fout[m],3); C_FIXDIV(Fout[m2],3);

         C_MUL(scratch[1],Fout[m] , *tw1);
         C_MUL(scratch[2],Fout[m2] , *tw2);

         C_ADD(scratch[3],scratch[1],scratch[2]);
         C_SUB(scratch[0],scratch[1],scratch[2]);
         tw1 += fstride;
         tw2 += fstride*2;

         Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
         Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

         C_MULBYSCALAR( scratch[0] , epi3.i );

         C_ADDTO(*Fout,scratch[3]);

         Fout[m2].r = Fout[m].r + scratch[0].i;
         Fout[m2].i = Fout[m].i - scratch[0].r;

         Fout[m].r -= scratch[0].i;
         Fout[m].i += scratch[0].r;

         ++Fout;
     }while(--k);
}

static void kf_bfly5(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    kiss_fft_cpx *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
    int u;
    kiss_fft_cpx scratch[13];
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx *tw;
    kiss_fft_cpx ya,yb;
    ya = twiddles[fstride*m];
    yb = twiddles[fstride*2*m];

    Fout0=Fout;
    Fout1=Fout0+m;
    Fout2=Fout0+2*m;
    Fout3=Fout0+3*m;
    Fout4=Fout0+4*m;

    tw=st->twiddles;
    for ( u=0; u<m; ++u ) {
        C_FIXDIV( *Fout0,5); C_FIXDIV( *Fout1,5); C_FIXDIV( *Fout2,5); C_FIXDIV( *Fout3,5); C_FIXDIV( *Fout4,5);
        scratch[0] = *Fout0;

        C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
        C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
        C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
        C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

        C_ADD( scratch[7],scratch[1],scratch[4]);
        C_SUB( scratch[10],scratch[1],scratch[4]);
        C_ADD( scratch[8],scratch[2],scratch[3]);
        C_SUB( scratch[9],scratch[2],scratch[3]);

        Fout0->r += scratch[7].r + scratch[8].r;
        Fout0->i += scratch[7].i + scratch[8].i;

        scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
        scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

        scratch[6].r =  S_MUL(scratch[10].i,ya.i) + S_MUL(scratch[9].i,yb.i);
        scratch[6].i = -S_MUL(scratch[10].r,ya.i) - S_MUL(scratch[9].r,yb.i);

        C_SUB(*Fout1,scratch[5],scratch[6]);
        C_ADD(*Fout4,scratch[5],scratch[6]);

        scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
        scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
        scratch[12].r = - S_MUL(scratch[10].i,yb.i) + S_MUL(scratch[9].i,ya.i);
        scratch[12].i = S_MUL(scratch[10].r,yb.i) - S_MUL(scratch[9].r,ya.i);

        C_ADD(*Fout2,scratch[11],scratch[12]);
        C_SUB(*Fout3,scratch[11],scratch[12]);

        ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
    }
}

#ifndef FIXED_POINT
static void kf_bfly7(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    /* radix-7 butterfly using the conjugate symmetry of the 7th roots of unity.
       out[k] = S_k - U_k, out[7-k] = S_k + U_k  (k=1..3), same scheme as kf_bfly5 */
    kiss_fft_cpx * tw = st->twiddles;
    const kiss_fft_scalar cc[4] = { 0, tw[fstride*m].r, tw[2*fstride*m].r, tw[3*fstride*m].r };
    const kiss_fft_scalar ss[4] = { 0, tw[fstride*m].i, tw[2*fstride*m].i, tw[3*fstride*m].i };
    int u, j, k;
    for (u = 0; u < m; ++u) {
        kiss_fft_cpx x[7], t[4], d[4], sk, uk;
        x[0] = Fout[u];
        C_FIXDIV(x[0],7);
        for (j = 1; j < 7; ++j) {
            kiss_fft_cpx v = Fout[u + j*m];
            C_FIXDIV(v,7);
            C_MUL(x[j], v, tw[j*u*fstride]);
        }
        for (j = 1; j <= 3; ++j) {
            C_ADD(t[j], x[j], x[7-j]);
            C_SUB(d[j], x[j], x[7-j]);
        }
        Fout[u].r = x[0].r + t[1].r + t[2].r + t[3].r;
        Fout[u].i = x[0].i + t[1].i + t[2].i + t[3].i;
        for (k = 1; k <= 3; ++k) {
            sk = x[0];
            uk.r = 0; uk.i = 0;
            for (j = 1; j <= 3; ++j) {
                int n = (k*j) % 7;
                int ci = n > 3 ? 7 - n : n;
                kiss_fft_scalar cs = cc[ci];
                kiss_fft_scalar sn = n > 3 ? -ss[ci] : ss[ci];
                sk.r += S_MUL(t[j].r, cs);
                sk.i += S_MUL(t[j].i, cs);
                uk.r += S_MUL(d[j].i, sn);
                uk.i -= S_MUL(d[j].r, sn);
            }
            C_SUB(Fout[u + k*m], sk, uk);
            C_ADD(Fout[u + (7-k)*m], sk, uk);
        }
    }
}
#endif /* FIXED_POINT */

/* perform the butterfly for one stage of a mixed radix FFT */
static void kf_bfly_generic(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m,
        int p
        )
{
    int u,k,q1,q;
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx t;
    int Norig = st->nfft;

    kiss_fft_cpx * scratch = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(sizeof(kiss_fft_cpx)*p);
    if (scratch == NULL){
        KISS_FFT_ERROR("Memory allocation failed.");
        return;
    }

    for ( u=0; u<m; ++u ) {
        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            scratch[q1] = Fout[ k  ];
            C_FIXDIV(scratch[q1],p);
            k += m;
        }

        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            int twidx=0;
            Fout[ k ] = scratch[0];
            for (q=1;q<p;++q ) {
                twidx += fstride * k;
                if (twidx>=Norig) twidx-=Norig;
                C_MUL(t,scratch[q] , twiddles[twidx] );
                C_ADDTO( Fout[ k ] ,t);
            }
            k += m;
        }
    }
    KISS_FFT_TMP_FREE(scratch);
}

/* Leaf stage (m==1): all twiddles are 1, read the p inputs directly from f
   (stride s) instead of copying them into Fout first. */
static void kf_leaf4(kiss_fft_cpx * Fout, const kiss_fft_cpx * f, const size_t s, const kiss_fft_cfg st)
{
    kiss_fft_cpx a = f[0], b = f[s], c = f[2*s], d = f[3*s];
    kiss_fft_cpx t5, t3, t4;
    C_FIXDIV(a,4); C_FIXDIV(b,4); C_FIXDIV(c,4); C_FIXDIV(d,4);
    t5.r = a.r - c.r; t5.i = a.i - c.i;
    a.r += c.r;       a.i += c.i;
    t3.r = b.r + d.r; t3.i = b.i + d.i;
    t4.r = b.r - d.r; t4.i = b.i - d.i;
    Fout[2].r = a.r - t3.r; Fout[2].i = a.i - t3.i;
    Fout[0].r = a.r + t3.r; Fout[0].i = a.i + t3.i;
    if (st->inverse) {
        Fout[1].r = t5.r - t4.i; Fout[1].i = t5.i + t4.r;
        Fout[3].r = t5.r + t4.i; Fout[3].i = t5.i - t4.r;
    } else {
        Fout[1].r = t5.r + t4.i; Fout[1].i = t5.i - t4.r;
        Fout[3].r = t5.r - t4.i; Fout[3].i = t5.i + t4.r;
    }
}

static void kf_leaf2(kiss_fft_cpx * Fout, const kiss_fft_cpx * f, const size_t s)
{
    kiss_fft_cpx a = f[0], b = f[s];
    C_FIXDIV(a,2); C_FIXDIV(b,2);
    C_ADD(Fout[0], a, b);
    C_SUB(Fout[1], a, b);
}

static void kf_leaf3(kiss_fft_cpx * Fout, const kiss_fft_cpx * f, const size_t s, const kiss_fft_cfg st, const size_t fstride)
{
    kiss_fft_cpx a = f[0], b = f[s], c = f[2*s];
    kiss_fft_cpx s3, s0;
    kiss_fft_cpx epi3 = st->twiddles[fstride];
    C_FIXDIV(a,3); C_FIXDIV(b,3); C_FIXDIV(c,3);
    C_ADD(s3,b,c);
    C_SUB(s0,b,c);
    Fout[1].r = a.r - HALF_OF(s3.r);
    Fout[1].i = a.i - HALF_OF(s3.i);
    C_MULBYSCALAR( s0 , epi3.i );
    C_ADD(Fout[0], a, s3);
    Fout[2].r = Fout[1].r + s0.i;
    Fout[2].i = Fout[1].i - s0.r;
    Fout[1].r -= s0.i;
    Fout[1].i += s0.r;
}

static void kf_leaf5(kiss_fft_cpx * Fout, const kiss_fft_cpx * f, const size_t s, const kiss_fft_cfg st, const size_t fstride)
{
    kiss_fft_cpx x0 = f[0], x1 = f[s], x2 = f[2*s], x3 = f[3*s], x4 = f[4*s];
    kiss_fft_cpx s7, s8, s9, s10, s5, s6, s11, s12;
    kiss_fft_cpx ya = st->twiddles[fstride], yb = st->twiddles[fstride*2];
    C_FIXDIV(x0,5); C_FIXDIV(x1,5); C_FIXDIV(x2,5); C_FIXDIV(x3,5); C_FIXDIV(x4,5);
    C_ADD( s7,x1,x4);
    C_SUB( s10,x1,x4);
    C_ADD( s8,x2,x3);
    C_SUB( s9,x2,x3);
    Fout[0].r = x0.r + s7.r + s8.r;
    Fout[0].i = x0.i + s7.i + s8.i;
    s5.r = x0.r + S_MUL(s7.r,ya.r) + S_MUL(s8.r,yb.r);
    s5.i = x0.i + S_MUL(s7.i,ya.r) + S_MUL(s8.i,yb.r);
    s6.r =  S_MUL(s10.i,ya.i) + S_MUL(s9.i,yb.i);
    s6.i = -S_MUL(s10.r,ya.i) - S_MUL(s9.r,yb.i);
    C_SUB(Fout[1],s5,s6);
    C_ADD(Fout[4],s5,s6);
    s11.r = x0.r + S_MUL(s7.r,yb.r) + S_MUL(s8.r,ya.r);
    s11.i = x0.i + S_MUL(s7.i,yb.r) + S_MUL(s8.i,ya.r);
    s12.r = - S_MUL(s10.i,yb.i) + S_MUL(s9.i,ya.i);
    s12.i = S_MUL(s10.r,yb.i) - S_MUL(s9.r,ya.i);
    C_ADD(Fout[2],s11,s12);
    C_SUB(Fout[3],s11,s12);
}

static
void kf_work(
        kiss_fft_cpx * Fout,
        const kiss_fft_cpx * f,
        const size_t fstride,
        int in_stride,
        int * factors,
        const kiss_fft_cfg st
        )
{
    kiss_fft_cpx * Fout_beg=Fout;
    const int p=*factors++; /* the radix  */
    const int m=*factors++; /* stage's fft length/p */
    const kiss_fft_cpx * Fout_end = Fout + p*m;

#ifdef _OPENMP
    // use openmp extensions at the
    // top-level (not recursive)
    if (fstride==1 && p<=5 && m!=1)
    {
        int k;

        // execute the p different work units in different threads
#       pragma omp parallel for
        for (k=0;k<p;++k)
            kf_work( Fout +k*m, f+ fstride*in_stride*k,fstride*p,in_stride,factors,st);
        // all threads have joined by this point

        switch (p) {
            case 2: kf_bfly2(Fout,fstride,st,m); break;
            case 3: kf_bfly3(Fout,fstride,st,m); break;
            case 4: kf_bfly4(Fout,fstride,st,m); break;
            case 5: kf_bfly5(Fout,fstride,st,m); break;
#ifndef FIXED_POINT
            case 7: kf_bfly7(Fout,fstride,st,m); break;
#endif
            default: kf_bfly_generic(Fout,fstride,st,m,p); break;
        }
        return;
    }
#endif

    if (m==1) {
        switch (p) {
            case 2: kf_leaf2(Fout,f,fstride*in_stride); return;
            case 3: kf_leaf3(Fout,f,fstride*in_stride,st,fstride); return;
            case 4: kf_leaf4(Fout,f,fstride*in_stride,st); return;
            case 5: kf_leaf5(Fout,f,fstride*in_stride,st,fstride); return;
            default: break;
        }
        do{
            *Fout = *f;
            f += fstride*in_stride;
        }while(++Fout != Fout_end );
    }else{
        do{
            // recursive call:
            // DFT of size m*p performed by doing
            // p instances of smaller DFTs of size m,
            // each one takes a decimated version of the input
            kf_work( Fout , f, fstride*p, in_stride, factors,st);
            f += fstride*in_stride;
        }while( (Fout += m) != Fout_end );
    }

    Fout=Fout_beg;

    // recombine the p smaller DFTs
    switch (p) {
        case 2: kf_bfly2(Fout,fstride,st,m); break;
        case 3: kf_bfly3(Fout,fstride,st,m); break;
        case 4: kf_bfly4(Fout,fstride,st,m); break;
        case 5: kf_bfly5(Fout,fstride,st,m); break;
#ifndef FIXED_POINT
            case 7: kf_bfly7(Fout,fstride,st,m); break;
#endif
        default: kf_bfly_generic(Fout,fstride,st,m,p); break;
    }
}

/*  facbuf is populated by p1,m1,p2,m2, ...
    where
    p[i] * m[i] = m[i-1]
    m0 = n                  */
static
void kf_factor(int n,int * facbuf)
{
    int p=4;
    double floor_sqrt;
    floor_sqrt = floor( sqrt((double)n) );

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;
                case 2: p = 3; break;
                default: p += 2; break;
            }
            if (p > floor_sqrt)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
    } while (n > 1);
}

/*
 *
 * User-callable function to allocate all necessary storage space for the fft.
 *
 * The return value is a contiguous block of memory, allocated with malloc.  As such,
 * It can be freed with free(), rather than a kiss_fft-specific function.
 * */
kiss_fft_cfg kiss_fft_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem )
{
    KISS_FFT_ALIGN_CHECK(mem)

    kiss_fft_cfg st=NULL;
    // reject empty/negative sizes and the overflow condition {memneeded > SIZE_MAX}.
    if (nfft <= 0 || (size_t)nfft >= (SIZE_MAX - 2*sizeof(struct kiss_fft_state))/sizeof(kiss_fft_cpx))
        return NULL;

    size_t memneeded = KISS_FFT_ALIGN_SIZE_UP(sizeof(struct kiss_fft_state)
        + sizeof(kiss_fft_cpx)*(nfft-1)); /* twiddle factors*/

    if ( lenmem==NULL ) {
        st = ( kiss_fft_cfg)KISS_FFT_MALLOC( memneeded );
    }else{
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }
    if (st) {
        int i;
        st->nfft=nfft;
        st->inverse = inverse_fft;

        for (i=0;i<nfft;++i) {
            const double pi=3.141592653589793238462643383279502884197169399375105820974944;
            double phase = -2*pi*i / nfft;
            if (st->inverse)
                phase *= -1;
            kf_cexp(st->twiddles+i, phase );
        }

        kf_factor(nfft,st->factors);
    }
    return st;
}


void kiss_fft_stride(kiss_fft_cfg st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int in_stride)
{
    if (fin == fout) {
        //NOTE: this is not really an in-place FFT algorithm.
        //It just performs an out-of-place FFT into a temp buffer
        if (fout == NULL){
            KISS_FFT_ERROR("fout buffer NULL.");
        return;
        }

        kiss_fft_cpx * tmpbuf = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC( sizeof(kiss_fft_cpx)*st->nfft);
        if (tmpbuf == NULL){
            KISS_FFT_ERROR("Memory allocation error.");
        return;
        }



        kf_work(tmpbuf,fin,1,in_stride, st->factors,st);
        memcpy(fout,tmpbuf,sizeof(kiss_fft_cpx)*st->nfft);
        KISS_FFT_TMP_FREE(tmpbuf);
    }else{
        kf_work( fout, fin, 1,in_stride, st->factors,st );
    }
}

void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
    kiss_fft_stride(cfg,fin,fout,1);
}


void kiss_fft_cleanup(void)
{
    // nothing needed any more
}

int kiss_fft_next_fast_size(int n)
{
    if (n <= 0) /* nothing to factor; avoid infinite loop for n<=0 */
        return 1;
    if (n > 2125764000) /* 2^5 * 3^12 * 5^3 is the largest 2,3,5-smooth int; n++ would overflow */
        return n;
    while(1) {
        int m=n;
        while ( (m%2) == 0 ) m/=2;
        while ( (m%3) == 0 ) m/=3;
        while ( (m%5) == 0 ) m/=5;
        if (m<=1)
            break; /* n is completely factorable by twos, threes, and fives */
        n++;
    }
    return n;
}
