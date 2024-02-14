/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
 
namespace af
{
 
/// Get the first non-zero dimension
static inline dim_type getFNSD(const int dim, af::dim4 dims)
{
    if(dim >= 0)
        return dim;

    dim_type fNSD = 0;
    for (dim_type i=0; i<4; ++i) {
        if (dims[i]>1) {
            fNSD = i;
            break;
        }
    }
    return fNSD;
}

}
