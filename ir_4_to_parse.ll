; ModuleID = "cfunc_wrapper"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare i32 @"_ZN8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64* %".1", {i8*, i32, i8*}** %".2", i64 %".3", i64 %".4")

define i64 @"cfunc._ZN8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64 %".1", i64 %".2")
{
entry:
  %".4" = alloca i64
  store i64 0, i64* %".4"
  store i64 0, i64* %".4"
  %"excinfo" = alloca {i8*, i32, i8*}*
  store {i8*, i32, i8*}* null, {i8*, i32, i8*}** %"excinfo"
  %".8" = call i32 @"_ZN8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64* %".4", {i8*, i32, i8*}** %"excinfo", i64 %".1", i64 %".2") noinline
  %".9" = load {i8*, i32, i8*}*, {i8*, i32, i8*}** %"excinfo"
  %".10" = icmp eq i32 %".8", 0
  %".11" = icmp eq i32 %".8", -2
  %".12" = icmp eq i32 %".8", -1
  %".13" = icmp eq i32 %".8", -3
  %".14" = or i1 %".10", %".11"
  %".15" = xor i1 %".14", -1
  %".16" = icmp sge i32 %".8", 1
  %".17" = select  i1 %".16", {i8*, i32, i8*}* %".9", {i8*, i32, i8*}* undef
  %".18" = load i64, i64* %".4"
  %".20" = alloca i32
  store i32 0, i32* %".20"
  br i1 %".15", label %"entry.if", label %"entry.endif", !prof !0
entry.if:
  call void @"numba_gil_ensure"(i32* %".20")
  br i1 %".16", label %"entry.if.if", label %"entry.if.endif"
entry.endif:
  ret i64 %".18"
.23:
  %".45" = call i8* @"PyUnicode_FromString"(i8* bitcast ([53 x i8]* @".const.<numba.core.cpu.CPUContext object at 0x7fb439b3b880>" to i8*))
  call void @"PyErr_WriteUnraisable"(i8* %".45")
  call void @"Py_DecRef"(i8* %".45")
  call void @"numba_gil_release"(i32* %".20")
  br label %"entry.endif"
entry.if.if:
  call void @"PyErr_Clear"()
  %".26" = load {i8*, i32, i8*}, {i8*, i32, i8*}* %".17"
  %".27" = extractvalue {i8*, i32, i8*} %".26", 0
  %".28" = load {i8*, i32, i8*}, {i8*, i32, i8*}* %".17"
  %".29" = extractvalue {i8*, i32, i8*} %".28", 1
  %".30" = load {i8*, i32, i8*}, {i8*, i32, i8*}* %".17"
  %".31" = extractvalue {i8*, i32, i8*} %".30", 2
  %".32" = call i8* @"numba_unpickle"(i8* %".27", i32 %".29", i8* %".31")
  %".33" = icmp ne i8* null, %".32"
  br i1 %".33", label %"entry.if.if.if", label %"entry.if.if.endif", !prof !1
entry.if.endif:
  br i1 %".13", label %"entry.if.endif.if", label %"entry.if.endif.endif"
entry.if.if.if:
  call void @"numba_do_raise"(i8* %".32")
  br label %"entry.if.if.endif"
entry.if.if.endif:
  br label %".23"
entry.if.endif.if:
  call void @"PyErr_SetNone"(i8* @"PyExc_StopIteration")
  br label %".23"
entry.if.endif.endif:
  br i1 %".12", label %"entry.if.endif.endif.if", label %"entry.if.endif.endif.endif"
entry.if.endif.endif.if:
  br label %".23"
entry.if.endif.endif.endif:
  call void @"PyErr_SetString"(i8* @"PyExc_SystemError", i8* bitcast ([43 x i8]* @".const.unknown error when calling native function" to i8*))
  br label %".23"
}

declare void @"numba_gil_ensure"(i32* %".1")

declare void @"PyErr_Clear"()

declare i8* @"numba_unpickle"(i8* %".1", i32 %".2", i8* %".3")

declare void @"numba_do_raise"(i8* %".1")

declare void @"PyErr_SetNone"(i8* %".1")

@"PyExc_StopIteration" = external global i8
declare void @"PyErr_SetString"(i8* %".1", i8* %".2")

@"PyExc_SystemError" = external global i8
@".const.unknown error when calling native function" = internal constant [43 x i8] c"unknown error when calling native function\00"
@".const.<numba.core.cpu.CPUContext object at 0x7fb439b3b880>" = internal constant [53 x i8] c"<numba.core.cpu.CPUContext object at 0x7fb439b3b880>\00"
declare i8* @"PyUnicode_FromString"(i8* %".1")

declare void @"PyErr_WriteUnraisable"(i8* %".1")

declare void @"Py_DecRef"(i8* %".1")

declare void @"numba_gil_release"(i32* %".1")

!0 = !{ !"branch_weights", i32 1, i32 99 }
!1 = !{ !"branch_weights", i32 99, i32 1 }