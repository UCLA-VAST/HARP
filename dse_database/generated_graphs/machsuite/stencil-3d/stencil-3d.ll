; ModuleID = 'stencil-3d.c'
source_filename = "stencil-3d.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @stencil3d(i64 %C0, i64 %C1, i64* %orig, i64* %sol) #0 {
entry:
  %C0.addr = alloca i64, align 8
  %C1.addr = alloca i64, align 8
  %orig.addr = alloca i64*, align 8
  %sol.addr = alloca i64*, align 8
  %sum0 = alloca i64, align 8
  %sum1 = alloca i64, align 8
  %mul0 = alloca i64, align 8
  %mul1 = alloca i64, align 8
  %i = alloca i64, align 8
  %j = alloca i64, align 8
  %ko = alloca i64, align 8
  %_in_ko = alloca i64, align 8
  store i64 %C0, i64* %C0.addr, align 8
  store i64 %C1, i64* %C1.addr, align 8
  store i64* %orig, i64** %orig.addr, align 8
  store i64* %sol, i64** %sol.addr, align 8
  store i64 1, i64* %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc70, %entry
  %0 = load i64, i64* %i, align 8
  %cmp = icmp slt i64 %0, 33
  br i1 %cmp, label %for.body, label %for.end72

for.body:                                         ; preds = %for.cond
  store i64 1, i64* %j, align 8
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc67, %for.body
  %1 = load i64, i64* %j, align 8
  %cmp2 = icmp slt i64 %1, 33
  br i1 %cmp2, label %for.body3, label %for.end69

for.body3:                                        ; preds = %for.cond1
  store i64 0, i64* %ko, align 8
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %2 = load i64, i64* %ko, align 8
  %cmp5 = icmp sle i64 %2, 31
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %3 = load i64, i64* %ko, align 8
  %mul = mul nsw i64 1, %3
  %add = add nsw i64 1, %mul
  store i64 %add, i64* %_in_ko, align 8
  %4 = load i64*, i64** %orig.addr, align 8
  %5 = load i64, i64* %_in_ko, align 8
  %add7 = add nsw i64 %5, 0
  %6 = load i64, i64* %j, align 8
  %7 = load i64, i64* %i, align 8
  %mul8 = mul nsw i64 34, %7
  %add9 = add nsw i64 %6, %mul8
  %mul10 = mul nsw i64 34, %add9
  %add11 = add nsw i64 %add7, %mul10
  %arrayidx = getelementptr inbounds i64, i64* %4, i64 %add11
  %8 = load i64, i64* %arrayidx, align 8
  store i64 %8, i64* %sum0, align 8
  %9 = load i64*, i64** %orig.addr, align 8
  %10 = load i64, i64* %_in_ko, align 8
  %add12 = add nsw i64 %10, 0
  %11 = load i64, i64* %j, align 8
  %12 = load i64, i64* %i, align 8
  %add13 = add nsw i64 %12, 1
  %mul14 = mul nsw i64 34, %add13
  %add15 = add nsw i64 %11, %mul14
  %mul16 = mul nsw i64 34, %add15
  %add17 = add nsw i64 %add12, %mul16
  %arrayidx18 = getelementptr inbounds i64, i64* %9, i64 %add17
  %13 = load i64, i64* %arrayidx18, align 8
  %14 = load i64*, i64** %orig.addr, align 8
  %15 = load i64, i64* %_in_ko, align 8
  %add19 = add nsw i64 %15, 0
  %16 = load i64, i64* %j, align 8
  %17 = load i64, i64* %i, align 8
  %sub = sub nsw i64 %17, 1
  %mul20 = mul nsw i64 34, %sub
  %add21 = add nsw i64 %16, %mul20
  %mul22 = mul nsw i64 34, %add21
  %add23 = add nsw i64 %add19, %mul22
  %arrayidx24 = getelementptr inbounds i64, i64* %14, i64 %add23
  %18 = load i64, i64* %arrayidx24, align 8
  %add25 = add nsw i64 %13, %18
  %19 = load i64*, i64** %orig.addr, align 8
  %20 = load i64, i64* %_in_ko, align 8
  %add26 = add nsw i64 %20, 0
  %21 = load i64, i64* %j, align 8
  %add27 = add nsw i64 %21, 1
  %22 = load i64, i64* %i, align 8
  %mul28 = mul nsw i64 34, %22
  %add29 = add nsw i64 %add27, %mul28
  %mul30 = mul nsw i64 34, %add29
  %add31 = add nsw i64 %add26, %mul30
  %arrayidx32 = getelementptr inbounds i64, i64* %19, i64 %add31
  %23 = load i64, i64* %arrayidx32, align 8
  %add33 = add nsw i64 %add25, %23
  %24 = load i64*, i64** %orig.addr, align 8
  %25 = load i64, i64* %_in_ko, align 8
  %add34 = add nsw i64 %25, 0
  %26 = load i64, i64* %j, align 8
  %sub35 = sub nsw i64 %26, 1
  %27 = load i64, i64* %i, align 8
  %mul36 = mul nsw i64 34, %27
  %add37 = add nsw i64 %sub35, %mul36
  %mul38 = mul nsw i64 34, %add37
  %add39 = add nsw i64 %add34, %mul38
  %arrayidx40 = getelementptr inbounds i64, i64* %24, i64 %add39
  %28 = load i64, i64* %arrayidx40, align 8
  %add41 = add nsw i64 %add33, %28
  %29 = load i64*, i64** %orig.addr, align 8
  %30 = load i64, i64* %_in_ko, align 8
  %add42 = add nsw i64 %30, 0
  %add43 = add nsw i64 %add42, 1
  %31 = load i64, i64* %j, align 8
  %32 = load i64, i64* %i, align 8
  %mul44 = mul nsw i64 34, %32
  %add45 = add nsw i64 %31, %mul44
  %mul46 = mul nsw i64 34, %add45
  %add47 = add nsw i64 %add43, %mul46
  %arrayidx48 = getelementptr inbounds i64, i64* %29, i64 %add47
  %33 = load i64, i64* %arrayidx48, align 8
  %add49 = add nsw i64 %add41, %33
  %34 = load i64*, i64** %orig.addr, align 8
  %35 = load i64, i64* %_in_ko, align 8
  %add50 = add nsw i64 %35, 0
  %sub51 = sub nsw i64 %add50, 1
  %36 = load i64, i64* %j, align 8
  %37 = load i64, i64* %i, align 8
  %mul52 = mul nsw i64 34, %37
  %add53 = add nsw i64 %36, %mul52
  %mul54 = mul nsw i64 34, %add53
  %add55 = add nsw i64 %sub51, %mul54
  %arrayidx56 = getelementptr inbounds i64, i64* %34, i64 %add55
  %38 = load i64, i64* %arrayidx56, align 8
  %add57 = add nsw i64 %add49, %38
  store i64 %add57, i64* %sum1, align 8
  %39 = load i64, i64* %sum0, align 8
  %40 = load i64, i64* %C0.addr, align 8
  %mul58 = mul nsw i64 %39, %40
  store i64 %mul58, i64* %mul0, align 8
  %41 = load i64, i64* %sum1, align 8
  %42 = load i64, i64* %C1.addr, align 8
  %mul59 = mul nsw i64 %41, %42
  store i64 %mul59, i64* %mul1, align 8
  %43 = load i64, i64* %mul0, align 8
  %44 = load i64, i64* %mul1, align 8
  %add60 = add nsw i64 %43, %44
  %45 = load i64*, i64** %sol.addr, align 8
  %46 = load i64, i64* %_in_ko, align 8
  %add61 = add nsw i64 %46, 0
  %47 = load i64, i64* %j, align 8
  %48 = load i64, i64* %i, align 8
  %mul62 = mul nsw i64 34, %48
  %add63 = add nsw i64 %47, %mul62
  %mul64 = mul nsw i64 34, %add63
  %add65 = add nsw i64 %add61, %mul64
  %arrayidx66 = getelementptr inbounds i64, i64* %45, i64 %add65
  store i64 %add60, i64* %arrayidx66, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %49 = load i64, i64* %ko, align 8
  %inc = add nsw i64 %49, 1
  store i64 %inc, i64* %ko, align 8
  br label %for.cond4, !llvm.loop !2

for.end:                                          ; preds = %for.cond4
  br label %for.inc67

for.inc67:                                        ; preds = %for.end
  %50 = load i64, i64* %j, align 8
  %inc68 = add nsw i64 %50, 1
  store i64 %inc68, i64* %j, align 8
  br label %for.cond1, !llvm.loop !4

for.end69:                                        ; preds = %for.cond1
  br label %for.inc70

for.inc70:                                        ; preds = %for.end69
  %51 = load i64, i64* %i, align 8
  %inc71 = add nsw i64 %51, 1
  store i64 %inc71, i64* %i, align 8
  br label %for.cond, !llvm.loop !5

for.end72:                                        ; preds = %for.cond
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project 1d6f08e61d9771baf5381198ac5d306f6cbcd302)"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.mustprogress"}
!4 = distinct !{!4, !3}
!5 = distinct !{!5, !3}
