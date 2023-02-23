; ModuleID = "wrapper"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare i32 @"_ZN8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64* %".1", {i8*, i32, i8*}** %".2", i64 %".3", i64 %".4")

define i8* @"_ZN7cpython8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i8* %"py_closure", i8* %"py_args", i8* %"py_kws")
{
entry:
  %".5" = alloca i8*
  %".6" = alloca i8*
  %".7" = call i32 (i8*, i8*, i64, i64, ...) @"PyArg_UnpackTuple"(i8* %"py_args", i8* bitcast ([2 x i8]* @".const.g" to i8*), i64 2, i64 2, i8** %".5", i8** %".6")
  %".8" = icmp eq i32 %".7", 0
  %".22" = alloca i64
  store i64 0, i64* %".22"
  %".38" = alloca i64
  store i64 0, i64* %".38"
  %".53" = alloca i64
  store i64 0, i64* %".53"
  %"excinfo" = alloca {i8*, i32, i8*}*
  store {i8*, i32, i8*}* null, {i8*, i32, i8*}** %"excinfo"
  %".72" = alloca i8*
  store i8* null, i8** %".72"
  br i1 %".8", label %"entry.if", label %"entry.endif", !prof !0
entry.if:
  ret i8* null
entry.endif:
  %".12" = load i8*, i8** @"_ZN08NumbaEnv8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"
  %".13" = ptrtoint i8* %".12" to i64
  %".14" = add i64 %".13", 16
  %".15" = inttoptr i64 %".14" to i8*
  %".16" = bitcast i8* %".15" to {i8*, i8*}*
  %".17" = icmp eq i8* null, %".12"
  br i1 %".17", label %"entry.endif.if", label %"entry.endif.endif", !prof !0
arg.end:
  ret i8* null
entry.endif.if:
  call void @"PyErr_SetString"(i8* @"PyExc_RuntimeError", i8* bitcast ([92 x i8]* @".const.missing Environment: _ZN08NumbaEnv8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx" to i8*))
  ret i8* null
entry.endif.endif:
  %".21" = load i8*, i8** %".5"
  %".24" = call i8* @"PyNumber_Long"(i8* %".21")
  %".25" = icmp ne i8* null, %".24"
  br i1 %".25", label %"entry.endif.endif.if", label %"entry.endif.endif.endif", !prof !1
entry.endif.endif.if:
  %".27" = call i64 @"PyLong_AsLongLong"(i8* %".24")
  call void @"Py_DecRef"(i8* %".24")
  store i64 %".27", i64* %".22"
  br label %"entry.endif.endif.endif"
entry.endif.endif.endif:
  %".31" = load i64, i64* %".22"
  %".32" = call i8* @"PyErr_Occurred"()
  %".33" = icmp ne i8* null, %".32"
  br i1 %".33", label %"entry.endif.endif.endif.if", label %"entry.endif.endif.endif.endif", !prof !0
entry.endif.endif.endif.if:
  br label %"arg.end"
entry.endif.endif.endif.endif:
  %".37" = load i8*, i8** %".6"
  %".40" = call i8* @"PyNumber_Long"(i8* %".37")
  %".41" = icmp ne i8* null, %".40"
  br i1 %".41", label %"entry.endif.endif.endif.endif.if", label %"entry.endif.endif.endif.endif.endif", !prof !1
arg0.err:
  br label %"arg.end"
entry.endif.endif.endif.endif.if:
  %".43" = call i64 @"PyLong_AsLongLong"(i8* %".40")
  call void @"Py_DecRef"(i8* %".40")
  store i64 %".43", i64* %".38"
  br label %"entry.endif.endif.endif.endif.endif"
entry.endif.endif.endif.endif.endif:
  %".47" = load i64, i64* %".38"
  %".48" = call i8* @"PyErr_Occurred"()
  %".49" = icmp ne i8* null, %".48"
  br i1 %".49", label %"entry.endif.endif.endif.endif.endif.if", label %"entry.endif.endif.endif.endif.endif.endif", !prof !0
entry.endif.endif.endif.endif.endif.if:
  br label %"arg0.err"
entry.endif.endif.endif.endif.endif.endif:
  store i64 0, i64* %".53"
  %".57" = call i32 @"_ZN8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64* %".53", {i8*, i32, i8*}** %"excinfo", i64 %".31", i64 %".47") noinline
  %".58" = load {i8*, i32, i8*}*, {i8*, i32, i8*}** %"excinfo"
  %".59" = icmp eq i32 %".57", 0
  %".60" = icmp eq i32 %".57", -2
  %".61" = icmp eq i32 %".57", -1
  %".62" = icmp eq i32 %".57", -3
  %".63" = or i1 %".59", %".60"
  %".64" = xor i1 %".63", -1
  %".65" = icmp sge i32 %".57", 1
  %".66" = select  i1 %".65", {i8*, i32, i8*}* %".58", {i8*, i32, i8*}* undef
  %".67" = load i64, i64* %".53"
  br i1 %".63", label %"entry.endif.endif.endif.endif.endif.endif.if", label %"entry.endif.endif.endif.endif.endif.endif.endif", !prof !1
arg1.err:
  br label %"arg0.err"
entry.endif.endif.endif.endif.endif.endif.if:
  br i1 %".60", label %"entry.endif.endif.endif.endif.endif.endif.if.if", label %"entry.endif.endif.endif.endif.endif.endif.if.endif"
entry.endif.endif.endif.endif.endif.endif.endif:
  br i1 %".65", label %"entry.endif.endif.endif.endif.endif.endif.endif.if", label %"entry.endif.endif.endif.endif.endif.endif.endif.endif"
entry.endif.endif.endif.endif.endif.endif.if.if:
  call void @"Py_IncRef"(i8* @"_Py_NoneStruct")
  ret i8* @"_Py_NoneStruct"
entry.endif.endif.endif.endif.endif.endif.if.endif:
  %".74" = call i8* @"PyLong_FromLongLong"(i64 %".67")
  store i8* %".74", i8** %".72"
  %".76" = load i8*, i8** %".72"
  ret i8* %".76"
.78:
  ret i8* null
entry.endif.endif.endif.endif.endif.endif.endif.if:
  call void @"PyErr_Clear"()
  %".81" = load {i8*, i32, i8*}, {i8*, i32, i8*}* %".66"
  %".82" = extractvalue {i8*, i32, i8*} %".81", 0
  %".83" = load {i8*, i32, i8*}, {i8*, i32, i8*}* %".66"
  %".84" = extractvalue {i8*, i32, i8*} %".83", 1
  %".85" = load {i8*, i32, i8*}, {i8*, i32, i8*}* %".66"
  %".86" = extractvalue {i8*, i32, i8*} %".85", 2
  %".87" = call i8* @"numba_unpickle"(i8* %".82", i32 %".84", i8* %".86")
  %".88" = icmp ne i8* null, %".87"
  br i1 %".88", label %"entry.endif.endif.endif.endif.endif.endif.endif.if.if", label %"entry.endif.endif.endif.endif.endif.endif.endif.if.endif", !prof !1
entry.endif.endif.endif.endif.endif.endif.endif.endif:
  br i1 %".62", label %"entry.endif.endif.endif.e...if", label %"entry.endif.endif.endif.e...endif"
entry.endif.endif.endif.endif.endif.endif.endif.if.if:
  call void @"numba_do_raise"(i8* %".87")
  br label %"entry.endif.endif.endif.endif.endif.endif.endif.if.endif"
entry.endif.endif.endif.endif.endif.endif.endif.if.endif:
  br label %".78"
entry.endif.endif.endif.e...if:
  call void @"PyErr_SetNone"(i8* @"PyExc_StopIteration")
  br label %".78"
entry.endif.endif.endif.e...endif:
  br i1 %".61", label %"entry.endif.endif.endif.e...endif.if", label %"entry.endif.endif.endif.e...endif.endif"
entry.endif.endif.endif.e...endif.if:
  br label %".78"
entry.endif.endif.endif.e...endif.endif:
  call void @"PyErr_SetString"(i8* @"PyExc_SystemError", i8* bitcast ([43 x i8]* @".const.unknown error when calling native function" to i8*))
  br label %".78"
}

declare i32 @"PyArg_UnpackTuple"(i8* %".1", i8* %".2", i64 %".3", i64 %".4", ...)

@".const.g" = internal constant [2 x i8] c"g\00"
@"_ZN08NumbaEnv8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx" = common global i8* null
declare void @"PyErr_SetString"(i8* %".1", i8* %".2")

@"PyExc_RuntimeError" = external global i8
@".const.missing Environment: _ZN08NumbaEnv8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx" = internal constant [92 x i8] c"missing Environment: _ZN08NumbaEnv8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx\00"
declare i8* @"PyNumber_Long"(i8* %".1")

declare i64 @"PyLong_AsLongLong"(i8* %".1")

declare void @"Py_DecRef"(i8* %".1")

declare i8* @"PyErr_Occurred"()

@"_Py_NoneStruct" = external global i8
declare void @"Py_IncRef"(i8* %".1")

declare i8* @"PyLong_FromLongLong"(i64 %".1")

declare void @"PyErr_Clear"()

declare i8* @"numba_unpickle"(i8* %".1", i32 %".2", i8* %".3")

declare void @"numba_do_raise"(i8* %".1")

declare void @"PyErr_SetNone"(i8* %".1")

@"PyExc_StopIteration" = external global i8
@"PyExc_SystemError" = external global i8
@".const.unknown error when calling native function" = internal constant [43 x i8] c"unknown error when calling native function\00"
!0 = !{ !"branch_weights", i32 1, i32 99 }
!1 = !{ !"branch_weights", i32 99, i32 1 }