# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, no-value-for-parameter
"""Iiconv2d schedule for ARM CPU."""

import tvm
from tvm import autotvm
from tvm import te
from tvm.topi.utils import simplify, traverse_inline
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
from ..utils import get_const_tuple
import random
import string

def intrin_iigemm_MxKxN(M, K, N, L, stride_w, itype, dtype):
    """Defines a SIMD-accelerated transposed matmul."""
    UNIQ_ID_LEN = 8
    uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))

    if isinstance(M, tvm.tir.IntImm):
        M = M.value
    if isinstance(K, tvm.tir.IntImm):
        K = K.value
    if isinstance(N, tvm.tir.IntImm):
        N = N.value

    assert itype == "uint8"
    assert dtype == "float32"
    
    A = te.placeholder(((M - 1) * stride_w + 1, K), name="a", dtype=dtype)
    B = te.placeholder((N, K), name="b", dtype=itype)
    R = te.placeholder((L,), name="r", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i * stride_w, k].astype(dtype) *
                            R[B[j, k].astype("int32")].astype(dtype), axis=k),
        name="c",
    )
    A_buf = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="A", offset_factor=1, strides=[te.var("A_s"), 1]
    )
    B_buf = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="B", offset_factor=1, strides=[te.var("B_s"), 1]
    )
    R_buf = tvm.tir.decl_buffer(
        R.shape, R.dtype, name="R", offset_factor=1, strides=[1]
    )
    C_buf = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="C", offset_factor=1, strides=[te.var("C_s"), 1]
    )

    def intrin_func(ins, outs):
        aa, bb, rr = ins
        cc = outs[0]

        def _reduce_update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"iigemm_{M}x{K}x{N}_update_{uniq_id}",
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    rr.access_ptr("r"),
                    cc.access_ptr("w"),
                    aa.strides[0] * stride_w,
                    bb.strides[0],
                    cc.strides[0]
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32", f"iigemm_{M}x{K}x{N}_reset_{uniq_id}", cc.access_ptr(
                        "w"), cc.strides[0]
                )
            )
            return ib.get()

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"iigemm_{M}x{K}x{N}_body_{uniq_id}",
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    rr.access_ptr("r"),
                    cc.access_ptr("w"),
                    aa.strides[0] * stride_w,
                    bb.strides[0],
                    cc.strides[0],
                )
            )
            return ib.get()

        return _body(), _reduce_reset(), _reduce_update()

    intrin_decl = te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: A_buf, B: B_buf, R: R_buf, C: C_buf})
    return intrin_decl, uniq_id


def iigemm_MxKxN_impl_lane2(M, K, N, L, uniq_id):
    """Emit C code for gemm impl."""
    # TODO(weberlo, areusch): are there any SIMD tricks to zero out arrays quickly?
    aa_pad_size = M * K
    bb_pad_size = N * K
    index_pad_size = L
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = f"""
#ifdef __cplusplus
extern "C"
#endif
#include <arm_neon.h>
#include <string.h>

int32_t iigemm_{M}x{K}x{N}_body_{uniq_id}(float32_t *A, uint8_t *B, float32_t *T, float32_t *C, int stride_A, int stride_B, int stride_C) {{
    float32x4_t aa[{K} / 4];
    uint8_t bb[{N} * {K} / 2];
    
    for (int i = 0; i < {N}; ++i) {{
        memcpy(bb + i * {K} / 2, B + i * stride_B, sizeof(uint8_t) * {K} / 2);
    }}

    for (int i = 0; i < {M}; ++i) {{
        for (int j = 0; j < {K}; j += 4) {{
            aa[j / 4] = vld1q_f32(A + i * stride_A + j);
        }}
        for (int j = 0; j < {N}; ++j) {{
            float32x4_t sum = vdupq_n_f32(0);
            uint8_t* l = bb + j * {K} / 2;
            for (int k = 0; k < {K} / 4; ++k) {{
                float32x4_t aa_vec = aa[k];
                float32x4_t bb_vec = vld1q_f32(T + (*l) * 2); ++l;
                float32x2_t bb_vec23 = vld1_f32(T + (*l) * 2); ++l;
                float32_t bb_vec2 = vget_lane_f32(bb_vec23, 0);
                float32_t bb_vec3 = vget_lane_f32(bb_vec23, 1);
                bb_vec = vsetq_lane_f32(bb_vec2, bb_vec, 2);
                bb_vec = vsetq_lane_f32(bb_vec3, bb_vec, 3);
                sum = vfmaq_f32(sum, aa_vec, bb_vec);
            }}
            C[i * stride_C + j] = vaddvq_f32(sum);
        }}
    }}
    return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
int32_t iigemm_{M}x{K}x{N}_update_{uniq_id}(float32_t *A, uint8_t *B, float32_t *T, float32_t *C, int stride_A, int stride_B, int stride_C) {{
    float32x4_t aa[{K} / 4];
    uint8_t bb[{N} * {K} / 2];

    for (int i = 0; i < {N}; ++i) {{
        memcpy(bb + i * {K} / 2, B + i * stride_B, sizeof(uint8_t) * {K} / 2);
    }}

    for (int i = 0; i < {M}; ++i) {{
        for (int j = 0; j < {K}; j += 4) {{
            aa[j / 4] = vld1q_f32(A + i * stride_A + j);
        }}
        for (int j = 0; j < {N}; ++j) {{
            float32x4_t sum = vdupq_n_f32(0);
            uint8_t* l = bb + j * {K} / 2;
            for (int k = 0; k < {K} / 4; ++k) {{
                float32x4_t aa_vec = aa[k];
                float32x4_t bb_vec = vld1q_f32(T + (*l) * 2); ++l;
                float32x2_t bb_vec23 = vld1_f32(T + (*l) * 2); ++l;
                float32_t bb_vec2 = vget_lane_f32(bb_vec23, 0);
                float32_t bb_vec3 = vget_lane_f32(bb_vec23, 1);
                bb_vec = vsetq_lane_f32(bb_vec2, bb_vec, 2);
                bb_vec = vsetq_lane_f32(bb_vec3, bb_vec, 3);
                sum = vfmaq_f32(sum, aa_vec, bb_vec);
            }}
            C[i * stride_C + j] = vaddvq_f32(sum);
        }}
    }}
    return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
int32_t iigemm_{M}x{K}x{N}_reset_{uniq_id}(int32_t *cc, int C_stride) {{
for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
    cc[i*C_stride + j] = 0;
    }}
}}
return 0;
}}

"""
    from tvm.contrib import utils, clang
    import os

    temp = utils.tempdir()
    ll_path = temp.relpath("temp" + str(uniq_id) + ".ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=[
                                "-march=armv8-a+simd", "-shared", "-fPIC", "-lm"],
                                cc=os.environ["TVM_NDK_CC"])

    return ll_code


def iigemm_MxKxN_impl_lane4(M, K, N, L, uniq_id):
    """Emit C code for gemm impl."""
    # TODO(weberlo, areusch): are there any SIMD tricks to zero out arrays quickly?
    aa_pad_size = M * K
    bb_pad_size = N * K
    index_pad_size = L
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = f"""
#ifdef __cplusplus
extern "C"
#endif
#include <arm_neon.h>
#include <string.h>

int32_t iigemm_{M}x{K}x{N}_body_{uniq_id}(float32_t *A, uint8_t *B, float32_t *T, float32_t *C, int stride_A, int stride_B, int stride_C) {{
    float32x4_t aa[{K} / 4];
    uint8_t bb[{N} * {K} / 4];
    
    for (int i = 0; i < {N}; ++i) {{
        memcpy(bb + i * {K} / 4, B + i * stride_B, sizeof(uint8_t) * {K} / 4);
    }}

    for (int i = 0; i < {M}; ++i) {{
        for (int j = 0; j < {K}; j += 4) {{
            aa[j / 4] = vld1q_f32(A + i * stride_A + j);
        }}
        for (int j = 0; j < {N}; ++j) {{
            float32x4_t sum = vdupq_n_f32(0);
            uint8_t* l = bb + j * {K} / 4;
            for (int k = 0; k < {K} / 4; ++k) {{
                float32x4_t aa_vec = aa[k];
                float32x4_t bb_vec = vld1q_f32(T + (*l) * 4); ++l;
                sum = vfmaq_f32(sum, aa_vec, bb_vec);
            }}
            C[i * stride_C + j] = vaddvq_f32(sum);
        }}
    }}
    return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
int32_t iigemm_{M}x{K}x{N}_update_{uniq_id}(float32_t *A, uint8_t *B, float32_t *T, float32_t *C, int stride_A, int stride_B, int stride_C) {{
    float32x4_t aa[{K} / 4];
    uint8_t bb[{N} * {K} / 4];

    for (int i = 0; i < {N}; ++i) {{
        memcpy(bb + i * {K} / 4, B + i * stride_B, sizeof(uint8_t) * {K} / 4);
    }}

    for (int i = 0; i < {M}; ++i) {{
        for (int j = 0; j < {K}; j += 4) {{
            aa[j / 4] = vld1q_f32(A + i * stride_A + j);
        }}
        for (int j = 0; j < {N}; ++j) {{
            float32x4_t sum = vdupq_n_f32(0);
            uint8_t* l = bb + j * {K} / 4;
            for (int k = 0; k < {K} / 4; ++k) {{
                float32x4_t aa_vec = aa[k];
                float32x4_t bb_vec = vld1q_f32(T + (*l) * 4); ++l;
                sum = vfmaq_f32(sum, aa_vec, bb_vec);
            }}
            C[i * stride_C + j] = vaddvq_f32(sum);
        }}
    }}
    return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
int32_t iigemm_{M}x{K}x{N}_reset_{uniq_id}(int32_t *cc, int C_stride) {{
for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
    cc[i*C_stride + j] = 0;
    }}
}}
return 0;
}}

"""
    from tvm.contrib import utils, clang
    import os

    temp = utils.tempdir()
    ll_path = temp.relpath("temp" + str(uniq_id) + ".ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=[
                                "-march=armv8-a+simd", "-shared", "-fPIC", "-lm"],
                                cc=os.environ["TVM_NDK_CC"])

    return ll_code


def iiconv2d_direct_simd_nhwc_compute(cfg, data, kernel, table, strides, padding, dilation, out_dtype):
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch_size, in_height, in_width, in_channels = data.shape
    kernel_h, kernel_w, out_channels, _ = kernel.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_height = simplify(
        (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w +
                        pad_left + pad_right) // stride_w + 1)

    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    padded_data = pad(data, pad_before, pad_after, name="padded_data")

    rc = te.reduce_axis((0, in_channels), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    assert out_channels % 4 == 0

    conv = te.compute(
        (batch_size, out_height, out_width, out_channels),
        lambda nn, yy, xx, ff: te.sum(
            padded_data[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ].astype(out_dtype)
            * table[0, kernel[ry, rx, ff, rc]].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        name="iiconv2d",
        tag=f"iiconv2d_nhwc_{stride_w}",
    )

    ###########################
    # Config Space Definition #
    ###########################
    n, oh, ow, co = (
        cfg.axis(batch_size.value),
        cfg.axis(out_height.value),
        cfg.axis(out_width.value),
        cfg.axis(out_channels.value),
    )
    kh, kw, ci = (
        cfg.reduce_axis(kernel_h.value),
        cfg.reduce_axis(kernel_w.value),
        cfg.reduce_axis(in_channels.value),
    )

    assert in_channels.value % 4 == 0
    owo, owi = cfg.define_split("tile_ow", ow, policy="factors", num_outputs=2)
    cio, cii = cfg.define_split(
        "tile_ci", ci, policy="factors", num_outputs=2, filter=lambda x: x.size[-1] % 4 == 0
    )
    coo, coi = cfg.define_split(
        "tile_co", co, policy="factors", num_outputs=2)

    cfg.define_reorder(
        "reorder_0_simd",
        [n, oh, owo, owi, coo, coi, kh, kw, cio, cii],
        policy="candidate",
        candidate=[
            [n, oh, owo, coo, cio, kh, kw, owi, coi, cii],
            [n, coo, oh, owo, cio, kh, kw, owi, coi, cii],
        ],
    )

    cfg.define_knob("auto_unroll_max_step", [0, 2, 4, 8, 16, 32])

    return conv


def iiconv2d_direct_simd_nhwc_schedule(cfg, outs):
    sched = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "iiconv2d_nhwc" not in op.tag:
            return

        stride_w = int(op.tag.split("_")[2])

        # extract tensors
        output = op.output(0)
        conv = op
        data_vec = conv.input_tensors[0]
        kernel = conv.input_tensors[1]  # pylint: disable=unused-variable
        table = conv.input_tensors[2]
        assert len(table.shape) == 2
        last = outs[0]  # pylint: disable=unused-variable

        # tile reduction axes
        n, oh, ow, co = sched[conv].op.axis
        kh, kw, ci = sched[conv].op.reduce_axis

        M = cfg["tile_ow"].size[-1]
        K = cfg["tile_ci"].size[-1]
        N = cfg["tile_co"].size[-1]
        _, L = get_const_tuple(table.shape)

        owo, owi = cfg["tile_ow"].apply(sched, conv, ow)
        cio, cii = cfg["tile_ci"].apply(sched, conv, ci)
        coo, coi = cfg["tile_co"].apply(sched, conv, co)

        cfg["reorder_0_simd"].apply(
            sched, conv, [n, oh, owo, owi, coo, coi, kh, kw, cio, cii])

        iigemm, uniq_id = intrin_iigemm_MxKxN(
            M, K, N, L, stride_w, kernel.dtype, output.dtype)

        sched[output].tensorize(owi, iigemm)

        if L / 256 == 2:
            impl = iigemm_MxKxN_impl_lane2
        else:
            impl = iigemm_MxKxN_impl_lane4
        sched[output].pragma(
            n, "import_llvm", impl(M, K, N, L, uniq_id))

        # this is the scope to attach global config inside this kernel
        kernel_scope = n

        # tune unroll
        sched[output].pragma(
            kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)

    traverse_inline(sched, outs[-1].op, _callback)
    return sched


@autotvm.register_topi_compute("iiconv2d_direct_simd.arm_cpu")
def iiconv2d_direct_simd(cfg, data, kernel, table, strides, padding, dilation, out_dtype):
    """Compute conv2d with SIMD (v7-a) """
    return iiconv2d_direct_simd_nhwc_compute(
        cfg, data, kernel, table, strides, padding, dilation, out_dtype
    )

@autotvm.register_topi_schedule("iiconv2d_direct_simd.arm_cpu")
def schedule_iiconv2d_direct_simd(cfg, outs):
    """Create schedule for iiconv2d_direct_simd"""
    return iiconv2d_direct_simd_nhwc_schedule(cfg, outs)


def group_iiconv2d_direct_simd_nhwc_compute(data, kernel, table, strides, padding, dilation, groups, out_dtype):
    """Group indirect index convolution operator in NHWC layout."""
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = get_const_tuple(data.shape)
    kernel_h, kernel_w, out_channel, _ = get_const_tuple(kernel.shape)

    assert in_channel % groups == 0, "input channels must divide group size"
    assert out_channel % groups == 0, "output channels must divide group size"

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding, (kernel_h, kernel_w))
    # compute the output shape
    out_height = simplify(
        (in_height - (kernel_h - 1) * dilation_h - 1 + pad_top + pad_down) // stride_h + 1
    )
    out_width = simplify(
        (in_width - (kernel_w - 1) * dilation_w - 1 + pad_left + pad_right) // stride_w + 1
    )
    # compute graph
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    temp = pad(data, pad_before, pad_after, name="pad_temp")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    rc = te.reduce_axis((0, in_channel // groups), name="rc")
    output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            temp[
                nn,
                yy * stride_h + ry * dilation_h,
                xx * stride_w + rx * dilation_w,
                ff // (out_channel // groups) * (in_channel // groups) + rc,
            ].astype(out_dtype)
            * table[kernel[ry, rx, ff, rc]].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        tag=f"group_iiconv2d_nhwc_{stride_w}",
    )

    return output


def group_iiconv2d_direct_simd_nhwc_schedule(cfg, outs):
    sched = te.create_schedule([x.op for x in outs])
    return sched


@autotvm.register_topi_compute("group_iiconv2d_direct_simd.arm_cpu")
def group_iiconv2d_direct_simd(cfg, data, kernel, table, strides, padding, dilation, out_dtype):
    """Compute iiconv2d with SIMD (v7-a) """
    return group_iiconv2d_direct_simd_nhwc_compute(
        cfg, data, kernel, table, strides, padding, dilation, out_dtype
    )


@autotvm.register_topi_schedule("group_iiconv2d_direct_simd.arm_cpu")
def schedule_group_iiconv2d_direct_simd(cfg, outs):
    """Create schedule for group_iiconv2d_direct_simd"""
    return group_iiconv2d_direct_simd_nhwc_schedule(cfg, outs)

