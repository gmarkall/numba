from numba.core.types import (
    b1,
    f2, f4, f8,
    i1, i2, i4, i8,
    u1, u2, u4, u8
)

# (b1, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64)
indices = (b1, u1, u2, u4, u8, i1, i2, i4, i8, f2, f4, f8)

ERR = None

promotion_matrix = (
    # b1
    (b1, u1,  u2,  u4,  u8,  i1,  i2,  i4,  i8,  f2, f4, f8),
    # u8
    (u1, u1,  u2,  u4,  u8,  ERR, ERR, ERR, ERR, f2, f4, f8),
    # u16
    (u2, u2,  u2,  u4,  u8,  ERR, ERR, ERR, ERR, f2, f4, f8),
    # u32
    (u4, u4,  u4,  u4,  u8,  ERR, ERR, ERR, ERR, f2, f4, f8),
    # u64
    (u8, u8,  u8,  u8,  u8,  ERR, ERR, ERR, ERR, f2, f4, f8),
    # i8
    (i1, ERR, ERR, ERR, ERR, i1,  i2,  i4,  i8,  f2, f4, f8),
    # i16
    (i2, ERR, ERR, ERR, ERR, i2,  i2,  i4,  i8,  f2, f4, f8),
    # i32
    (i4, ERR, ERR, ERR, ERR, i4,  i4,  i4,  i8,  f2, f4, f8),
    # i64
    (i8, ERR, ERR, ERR, ERR, i8,  i8,  i8,  i8,  f2, f4, f8),
    # f16
    (f2, f2,  f2,  f2,  f2,  f2,  f2,  f2,  f2,  f2, f4, f8),
    # f32
    (f4, f4,  f4,  f4,  f4,  f4,  f4,  f4,  f4,  f4, f4, f8),
    # f64
    (f8, f8,  f8,  f8,  f8,  f8,  f8,  f8,  f8,  f8, f8, f8),
)

PROMOTE_TYPES = {}

for i, t1 in enumerate(indices):
    for j, t2 in enumerate(indices):
        PROMOTE_TYPES[t1, t2] = promotion_matrix[i][j]
