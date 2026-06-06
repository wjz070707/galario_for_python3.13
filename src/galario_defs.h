/******************************************************************************
* This file is part of GALARIO:                                               *
* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
*                                                                             *
* Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  *
* Copyright (C) 2026, wjz070707.                                             *
*                                                                             *
* This program is free software: you can redistribute it and/or modify        *
* it under the terms of the Lesser GNU General Public License as published by *
* the Free Software Foundation, either version 3 of the License, or           *
* (at your option) any later version.                                         *
*                                                                             *
* This program is distributed in the hope that it will be useful,             *
* but WITHOUT ANY WARRANTY; without even the implied warranty of              *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        *
*                                                                             *
* For more details see the LICENSE file.                                      *
* Maintained at https://github.com/wjz070707/galario_for_python3.13           *
******************************************************************************/

#pragma once

#ifdef __CUDACC__
    #include <cufft.h>
#else
    #include <complex>
#endif

typedef double dreal;

#ifdef __CUDACC__
    typedef cufftDoubleComplex dcomplex;
#else
    typedef std::complex<dreal> dcomplex;
#endif
