/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if CPLX
#define set(a, b) a = b
#define set_scalar(a, b) do {                   \
        a.x = b;                                \
        a.y = 0;                                \
    } while(0)

Ty mul(Ty a, Tp b) { a.x = a.x * b; a.y = a.y * b; return a; }
Ty div(Ty a, Tp b) { a.x = a.x / b; a.y = a.y / b; return a; }

#else

#define set(a, b) a = b
#define set_scalar(a, b) a = b
#define mul(a, b) ((a) * (b))
#define div(a, b) ((a) / (b))

#endif

void transform_n(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const dim_type xido, const dim_type yido, const dim_type nimages)
{
    // Compute input index
    const dim_type xidi = round(xido * tmat[0]
                              + yido * tmat[1]
                                     + tmat[2]);
    const dim_type yidi = round(xido * tmat[3]
                              + yido * tmat[4]
                                     + tmat[5]);

    // Compute memory location of indices
    const dim_type loci = yidi * in.strides[1]  + xidi;
    const dim_type loco = yido * out.strides[1] + xido;

    for(int i = 0; i < nimages; i++) {
        // Compute memory location of indices
        dim_type ioff = loci + i * in.strides[2];
        dim_type ooff = loco + i * out.strides[2];

        T val; set_scalar(val, 0);
        if (xidi < in.dims[0] && yidi < in.dims[1] && xidi >= 0 && yidi >= 0) val = d_in[ioff];

        d_out[ooff] = val;
    }
}

void transform_b(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const dim_type xido, const dim_type yido, const dim_type nimages)
{
    const dim_type loco = (yido * out.strides[1] + xido);

    // Compute input index
    const float xid = xido * tmat[0]
                    + yido * tmat[1]
                           + tmat[2];
    const float yid = xido * tmat[3]
                    + yido * tmat[4]
                           + tmat[5];

    T zero; set_scalar(zero, 0);
    if (xid < 0 || yid < 0 || in.dims[0] < xid || in.dims[1] < yid) {
        for(int i = 0; i < nimages; i++) {
            set(d_out[loco + i * out.strides[2]], zero);
        }
        return;
    }

    const float grd_x = floor(xid),  grd_y = floor(yid);
    const float off_x = xid - grd_x, off_y = yid - grd_y;

    // Check if pVal and pVal + 1 are both valid indices
    const bool condY = (yid < in.dims[1] - 1);
    const bool condX = (xid < in.dims[0] - 1);

    // Compute weights used
    const float wt00 = (1.0 - off_x) * (1.0 - off_y);
    const float wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
    const float wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
    const float wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

    const float wt = wt00 + wt10 + wt01 + wt11;

    const dim_type loci = grd_y * in.strides[1] + grd_x;
    for(int i = 0; i < nimages; i++) {
        const dim_type ioff = loci + (i * in.strides[2]);
        const dim_type ooff = loco + (i * out.strides[2]);

        // Compute Weighted Values
        T v00 =                    wt00 * d_in[ioff];
        T v10 = (condY) ?          wt10 * d_in[ioff + in.strides[1]]     : zero;
        T v01 = (condX) ?          wt01 * d_in[ioff + 1]                 : zero;
        T v11 = (condX && condY) ? wt11 * d_in[ioff + in.strides[1] + 1] : zero;
        T vo = v00 + v10 + v01 + v11;

        d_out[ooff] = (vo / wt);
    }
}
