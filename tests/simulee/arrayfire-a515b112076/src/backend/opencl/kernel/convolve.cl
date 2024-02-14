/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

dim_type index(dim_type i, dim_type j, dim_type k, dim_type jstride, dim_type kstride)
{
    return i+j*jstride+k*kstride;
}

#if BASE_DIM==1
kernel
void convolve(global T *out, KParam oInfo, global T const *signal, KParam sInfo,
              local T *localMem, constant accType const *impulse, KParam fInfo,
              dim_type nBBS0, dim_type nBBS1, dim_type ostep1, dim_type ostep2,
              dim_type ostep3, dim_type sstep1, dim_type sstep2, dim_type sstep3)
{
    dim_type fLen     = fInfo.dims[0];
    dim_type padding  = fLen-1;
    dim_type shrdLen  = get_local_size(0) + 2*padding;
    const unsigned b1 = get_group_id(0)/nBBS0;
    const unsigned b0 = get_group_id(0)-nBBS0*b1;
    const unsigned b3 = get_group_id(1)/nBBS1;
    const unsigned b2 = get_group_id(1)-nBBS1*b3;

    global T *dst = out + (b1 * oInfo.strides[1] +  /* activated with batched input signal */
                       ostep1 * oInfo.strides[1] +  /* activated with batched input filter */
                           b2 * oInfo.strides[2] +  /* activated with batched input signal */
                       ostep2 * oInfo.strides[2] +  /* activated with batched input filter */
                           b3 * oInfo.strides[3] +  /* activated with batched input signal */
                       ostep3 * oInfo.strides[3]);  /* activated with batched input filter */

    global T const *src = signal + sInfo.offset + (b1 * sInfo.strides[1] + /* activated with batched input signal */
                                               sstep1 * sInfo.strides[1] + /* activated with batched input filter */
                                                   b2 * sInfo.strides[2] + /* activated with batched input signal */
                                               sstep2 * sInfo.strides[2] + /* activated with batched input filter */
                                                   b3 * sInfo.strides[3] + /* activated with batched input signal */
                                               sstep3 * sInfo.strides[3]); /* activated with batched input filter */

    dim_type gx  = get_local_size(0)*b0;

    for (dim_type i=get_local_id(0); i<shrdLen; i+=get_local_size(0)) {
        dim_type idx = gx-padding + i;
        localMem[i]  = (idx>=0 && idx<sInfo.dims[0]) ? src[idx*sInfo.strides[0]] : (T)(0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    gx += get_local_id(0);

    if (gx>=0 && gx<oInfo.dims[0]) {
        dim_type lx   = get_local_id(0) + padding + (EXPAND ? 0 : fLen>>1);
        accType accum = (accType)(0);
        for(dim_type f=0; f<fLen; ++f) {
            accum = accum + ((accType)localMem[lx-f] * (accType)impulse[f]);
        }
        dst[gx] = (T)accum;
    }
}
#endif

#if BASE_DIM==2
kernel
void convolve(global T *out, KParam oInfo, global T const *signal, KParam sInfo,
              constant accType const *impulse, KParam fInfo,
              dim_type nBBS0, dim_type nBBS1, dim_type ostep2,
              dim_type ostep3, dim_type sstep2, dim_type sstep3)
{
    local T localMem[C_SIZE];

    dim_type radius0  = FLEN0-1;
    dim_type radius1  = FLEN1-1;
    dim_type padding0 = 2*radius0;
    dim_type padding1 = 2*radius1;
    dim_type shrdLen0 = get_local_size(0) + padding0;
    dim_type shrdLen1 = get_local_size(1) + padding1;

    unsigned b0  = get_group_id(0)/nBBS0;
    unsigned b1  = get_group_id(1)/nBBS1;

    global T *dst = out + (b0 * oInfo.strides[2] + /* activated with batched input signal */
                       ostep2 * oInfo.strides[2] + /* activated with batched input filter */
                           b1 * oInfo.strides[3] + /* activated with batched input signal */
                       ostep3 * oInfo.strides[3]); /* activated with batched input filter */

    global const T *src = signal + sInfo.offset + (b0 * sInfo.strides[2] + /* activated with batched input signal */
                                               sstep2 * sInfo.strides[2] + /* activated with batched input filter */
                                                   b1 * sInfo.strides[3] + /* activated with batched input signal */
                                               sstep3 * sInfo.strides[3]); /* activated with batched input filter */

    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);
    dim_type gx = get_local_size(0) * (get_group_id(0)-b0*nBBS0) + lx;
    dim_type gy = get_local_size(1) * (get_group_id(1)-b1*nBBS1) + ly;

    // below loops are traditional loops, they only run multiple
    // times filter length is more than launch size
    dim_type s0 = sInfo.strides[0];
    dim_type s1 = sInfo.strides[1];
    dim_type d0 = sInfo.dims[0];
    dim_type d1 = sInfo.dims[1];
    for (dim_type b=ly, gy2=gy; b<shrdLen1; b+=get_local_size(1), gy2+=get_local_size(1)) {
        dim_type j = gy2-radius1;
        bool is_j  = j>=0 && j<d1;
        // move row_set get_local_size(1) along coloumns
        for (dim_type a=lx, gx2=gx; a<shrdLen0; a+=get_local_size(0), gx2+=get_local_size(0)) {
            dim_type i = gx2-radius0;
            bool is_i  = i>=0 && i<d0;
            localMem[b*shrdLen0+a] = (is_i && is_j ? src[i*s0+j*s1] : (T)(0));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx<oInfo.dims[0] && gy<oInfo.dims[1]) {
        dim_type ci = lx + radius0 + (EXPAND ? 0 : FLEN0>>1);
        dim_type cj = ly + radius1 + (EXPAND ? 0 : FLEN1>>1);

        accType accum = (accType)(0);
        for(dim_type fj=0; fj<FLEN1; ++fj) {
            for(dim_type fi=0; fi<FLEN0; ++fi) {
                accType f_val = impulse[fj*FLEN0+fi];
                T s_val = localMem[(cj-fj)*shrdLen0+(ci-fi)];
                accum   = accum + ((accType)s_val*(accType)f_val);
            }
        }
        dst[gy*oInfo.strides[1]+gx] = (T)accum;
    }
}
#endif

#if BASE_DIM==3
kernel
void convolve(global T *out, KParam oInfo, global T const *signal, KParam sInfo,
              local T *localMem, constant accType const *impulse, KParam fInfo,
              dim_type nBBS0, dim_type nBBS1, dim_type o1, dim_type ostep2,
              dim_type ostep3, dim_type sstep1, dim_type sstep2, dim_type sstep3)
{
    dim_type fLen0    = fInfo.dims[0];
    dim_type fLen1    = fInfo.dims[1];
    dim_type fLen2    = fInfo.dims[2];
    dim_type radius0  = fLen0-1;
    dim_type radius1  = fLen1-1;
    dim_type radius2  = fLen2-1;
    dim_type shrdLen0 = get_local_size(0) + 2*radius0;
    dim_type shrdLen1 = get_local_size(1) + 2*radius1;
    dim_type shrdLen2 = get_local_size(2) + 2*radius2;
    dim_type skStride = shrdLen0 * shrdLen1;
    dim_type fStride  = fLen0 * fLen1;
    unsigned b2  = get_group_id(0)/nBBS0;

    global T *dst = out + (b2 * oInfo.strides[3] + /* activated with batched input signal */
                       ostep3 * oInfo.strides[3]); /* activated with batched input filter */

    global const T *src = signal + sInfo.offset + (b2 * sInfo.strides[3] + /* activated with batched input signal */
                                               sstep3 * sInfo.strides[3]); /* activated with batched input filter */

    dim_type lx  = get_local_id(0);
    dim_type ly  = get_local_id(1);
    dim_type lz  = get_local_id(2);
    dim_type gx  = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    dim_type gy  = get_local_size(1) * get_group_id(1) + ly;
    dim_type gz  = get_local_size(2) * get_group_id(2) + lz;
    dim_type lx2 = lx + get_local_size(0);
    dim_type ly2 = ly + get_local_size(1);
    dim_type lz2 = lz + get_local_size(2);
    dim_type gx2 = gx + get_local_size(0);
    dim_type gy2 = gy + get_local_size(1);
    dim_type gz2 = gz + get_local_size(2);

    dim_type s0 = sInfo.strides[0];
    dim_type s1 = sInfo.strides[1];
    dim_type s2 = sInfo.strides[2];
    dim_type d0 = sInfo.dims[0];
    dim_type d1 = sInfo.dims[1];
    dim_type d2 = sInfo.dims[2];

    for (dim_type c=lz, gz2=gz; c<shrdLen2; c+=get_local_size(2), gz2+=get_local_size(2)) {
        dim_type k = gz2-radius2;
        bool is_k  = k>=0 && k<d2;
        for (dim_type b=ly, gy2=gy; b<shrdLen1; b+=get_local_size(1), gy2+=get_local_size(1)) {
            dim_type j = gy2-radius1;
            bool is_j  = j>=0 && j<d1;
            for (dim_type a=lx, gx2=gx; a<shrdLen0; a+=get_local_size(0), gx2+=get_local_size(0)) {
                dim_type i = gx2-radius0;
                bool is_i  = i>=0 && i<d0;
                localMem[c*skStride+b*shrdLen0+a] = (is_i && is_j && is_k ? src[i*s0+j*s1+k*s2] : (T)(0));
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx<oInfo.dims[0] && gy<oInfo.dims[1] && gz<oInfo.dims[2]) {
        dim_type ci = lx + radius0 + (EXPAND ? 0 : fLen0>>1);
        dim_type cj = ly + radius1 + (EXPAND ? 0 : fLen1>>1);
        dim_type ck = lz + radius2 + (EXPAND ? 0 : fLen2>>1);

        accType accum = (accType)(0);
        for(dim_type fk=0; fk<fLen2; ++fk) {
            for(dim_type fj=0; fj<fLen1; ++fj) {
                for(dim_type fi=0; fi<fLen0; ++fi) {
                    accType f_val = impulse[index(fi, fj, fk, fLen0, fStride)];
                    T s_val = localMem[index(ci-fi, cj-fj, ck-fk, shrdLen0, skStride)];
                    accum   = accum + ((accType)s_val*(accType)f_val);
                }
            }
        }
        dst[index(gx, gy, gz, oInfo.strides[1], oInfo.strides[2])] = (T)accum;
    }
}
#endif
