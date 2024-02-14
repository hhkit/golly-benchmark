/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <iostream>
#include "defines.h"

namespace af {

class AFAPI exception
{
    public:
    exception();
    exception(const char *msg);
    exception(const char *file, unsigned line);
    exception(const char *file, unsigned line, af_err err);
    exception(const char *msg, const char *file, unsigned line, af_err err);
    virtual ~exception() throw() {}
    virtual const char *what() const throw() { return m_msg; }

    char m_msg[1024];

    friend inline std::ostream& operator<<(std::ostream &s, const exception &e) { return s << e.what(); }
};

}
