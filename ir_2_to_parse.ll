; ModuleID = "g$2"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@"_ZN08NumbaEnv8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx" = common global i8* null
define i32 @"_ZN8__main__1gB2v2B36c8tJTIcFHzwl2ILiXkcBV0IBS2sCAA_3d_3dExx"(i64* noalias nocapture %"retptr", {i8*, i32, i8*}** noalias nocapture %"excinfo", i64 %"arg.a", i64 %"arg.b")
{
entry:
  br label %"B0"
B0:
  %".6" = add nsw i64 %"arg.a", %"arg.b"
  store i64 %".6", i64* %"retptr"
  ret i32 0
}
