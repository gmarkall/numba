; ModuleID = "f$1"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@"_ZN08NumbaEnv8__main__1fB2v1B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx" = common global i8* null
define i32 @"_ZN8__main__1fB2v1B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64* noalias nocapture %"retptr", {i8*, i32, i8*}** noalias nocapture %"excinfo", i64 %"arg.a", i64 %"arg.b")
{
entry:
  %".7" = alloca i64
  store i64 0, i64* %".7"
  %"excinfo.1" = alloca {i8*, i32, i8*}*
  store {i8*, i32, i8*}* null, {i8*, i32, i8*}** %"excinfo.1"
  %"try_state" = alloca i64
  store i64 0, i64* %"try_state"
  br label %"B0"
B0:
  %".6" = load i8*, i8** @"numba.dynamic.globals.7fb43a529260"
  store i64 0, i64* %".7"
  %".11" = call i32 @"_ZN8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64* %".7", {i8*, i32, i8*}** %"excinfo.1", i64 %"arg.a", i64 %"arg.b")
  %".12" = load {i8*, i32, i8*}*, {i8*, i32, i8*}** %"excinfo.1"
  %".13" = icmp eq i32 %".11", 0
  %".14" = icmp eq i32 %".11", -2
  %".15" = icmp eq i32 %".11", -1
  %".16" = icmp eq i32 %".11", -3
  %".17" = or i1 %".13", %".14"
  %".18" = xor i1 %".17", -1
  %".19" = icmp sge i32 %".11", 1
  %".20" = select  i1 %".19", {i8*, i32, i8*}* %".12", {i8*, i32, i8*}* undef
  %".21" = load i64, i64* %".7"
  br i1 %".18", label %"B0.if", label %"B0.endif", !prof !0
B0.if:
  store i64 0, i64* %"try_state"
  %".25" = load i64, i64* %"try_state"
  %".26" = icmp ugt i64 %".25", 0
  %".27" = load {i8*, i32, i8*}*, {i8*, i32, i8*}** %"excinfo"
  store {i8*, i32, i8*}* %".20", {i8*, i32, i8*}** %"excinfo"
  %".29" = xor i1 %".26", -1
  br i1 %".29", label %"B0.if.if", label %"B0.if.endif"
B0.endif:
  store i64 %".21", i64* %"retptr"
  ret i32 0
B0.if.if:
  ret i32 %".11", !ret_is_raise !1
B0.if.endif:
  br label %"B0.endif"
}

@"numba.dynamic.globals.7fb43a529260" = linkonce global i8* inttoptr (i64 140412049330784 to i8*)
declare i32 @"_ZN8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64* noalias nocapture %"retptr", {i8*, i32, i8*}** noalias nocapture %"excinfo", i64 %"arg.a", i64 %"arg.b")

!0 = !{ !"branch_weights", i32 1, i32 99 }
!1 = !{ i1 1 }