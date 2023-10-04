; ModuleID = 'syr2k.c'
source_filename = "syr2k.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @kernel_syr2k(double %alpha, double %beta, [80 x double]* %C, [60 x double]* %A, [60 x double]* %B) #0 {
entry:
  %alpha.addr = alloca double, align 8
  %beta.addr = alloca double, align 8
  %C.addr = alloca [80 x double]*, align 8
  %A.addr = alloca [60 x double]*, align 8
  %B.addr = alloca [60 x double]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store double %alpha, double* %alpha.addr, align 8
  store double %beta, double* %beta.addr, align 8
  store [80 x double]* %C, [80 x double]** %C.addr, align 8
  store [60 x double]* %A, [60 x double]** %A.addr, align 8
  store [60 x double]* %B, [60 x double]** %B.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc47, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 80
  br i1 %cmp, label %for.body, label %for.end49

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 80
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load i32, i32* %j, align 4
  %3 = load i32, i32* %i, align 4
  %cmp4 = icmp sle i32 %2, %3
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %for.body3
  %4 = load double, double* %beta.addr, align 8
  %5 = load [80 x double]*, [80 x double]** %C.addr, align 8
  %6 = load i32, i32* %i, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds [80 x double], [80 x double]* %5, i64 %idxprom
  %7 = load i32, i32* %j, align 4
  %idxprom5 = sext i32 %7 to i64
  %arrayidx6 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx, i64 0, i64 %idxprom5
  %8 = load double, double* %arrayidx6, align 8
  %mul = fmul double %8, %4
  store double %mul, double* %arrayidx6, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body3
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %9 = load i32, i32* %j, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond1, !llvm.loop !2

for.end:                                          ; preds = %for.cond1
  store i32 0, i32* %k, align 4
  br label %for.cond7

for.cond7:                                        ; preds = %for.inc44, %for.end
  %10 = load i32, i32* %k, align 4
  %cmp8 = icmp slt i32 %10, 60
  br i1 %cmp8, label %for.body9, label %for.end46

for.body9:                                        ; preds = %for.cond7
  store i32 0, i32* %j, align 4
  br label %for.cond10

for.cond10:                                       ; preds = %for.inc41, %for.body9
  %11 = load i32, i32* %j, align 4
  %cmp11 = icmp slt i32 %11, 80
  br i1 %cmp11, label %for.body12, label %for.end43

for.body12:                                       ; preds = %for.cond10
  %12 = load i32, i32* %j, align 4
  %13 = load i32, i32* %i, align 4
  %cmp13 = icmp sle i32 %12, %13
  br i1 %cmp13, label %if.then14, label %if.end40

if.then14:                                        ; preds = %for.body12
  %14 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %15 = load i32, i32* %j, align 4
  %idxprom15 = sext i32 %15 to i64
  %arrayidx16 = getelementptr inbounds [60 x double], [60 x double]* %14, i64 %idxprom15
  %16 = load i32, i32* %k, align 4
  %idxprom17 = sext i32 %16 to i64
  %arrayidx18 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx16, i64 0, i64 %idxprom17
  %17 = load double, double* %arrayidx18, align 8
  %18 = load double, double* %alpha.addr, align 8
  %mul19 = fmul double %17, %18
  %19 = load [60 x double]*, [60 x double]** %B.addr, align 8
  %20 = load i32, i32* %i, align 4
  %idxprom20 = sext i32 %20 to i64
  %arrayidx21 = getelementptr inbounds [60 x double], [60 x double]* %19, i64 %idxprom20
  %21 = load i32, i32* %k, align 4
  %idxprom22 = sext i32 %21 to i64
  %arrayidx23 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx21, i64 0, i64 %idxprom22
  %22 = load double, double* %arrayidx23, align 8
  %mul24 = fmul double %mul19, %22
  %23 = load [60 x double]*, [60 x double]** %B.addr, align 8
  %24 = load i32, i32* %j, align 4
  %idxprom25 = sext i32 %24 to i64
  %arrayidx26 = getelementptr inbounds [60 x double], [60 x double]* %23, i64 %idxprom25
  %25 = load i32, i32* %k, align 4
  %idxprom27 = sext i32 %25 to i64
  %arrayidx28 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx26, i64 0, i64 %idxprom27
  %26 = load double, double* %arrayidx28, align 8
  %27 = load double, double* %alpha.addr, align 8
  %mul29 = fmul double %26, %27
  %28 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %29 = load i32, i32* %i, align 4
  %idxprom30 = sext i32 %29 to i64
  %arrayidx31 = getelementptr inbounds [60 x double], [60 x double]* %28, i64 %idxprom30
  %30 = load i32, i32* %k, align 4
  %idxprom32 = sext i32 %30 to i64
  %arrayidx33 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx31, i64 0, i64 %idxprom32
  %31 = load double, double* %arrayidx33, align 8
  %mul34 = fmul double %mul29, %31
  %add = fadd double %mul24, %mul34
  %32 = load [80 x double]*, [80 x double]** %C.addr, align 8
  %33 = load i32, i32* %i, align 4
  %idxprom35 = sext i32 %33 to i64
  %arrayidx36 = getelementptr inbounds [80 x double], [80 x double]* %32, i64 %idxprom35
  %34 = load i32, i32* %j, align 4
  %idxprom37 = sext i32 %34 to i64
  %arrayidx38 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx36, i64 0, i64 %idxprom37
  %35 = load double, double* %arrayidx38, align 8
  %add39 = fadd double %35, %add
  store double %add39, double* %arrayidx38, align 8
  br label %if.end40

if.end40:                                         ; preds = %if.then14, %for.body12
  br label %for.inc41

for.inc41:                                        ; preds = %if.end40
  %36 = load i32, i32* %j, align 4
  %inc42 = add nsw i32 %36, 1
  store i32 %inc42, i32* %j, align 4
  br label %for.cond10, !llvm.loop !4

for.end43:                                        ; preds = %for.cond10
  br label %for.inc44

for.inc44:                                        ; preds = %for.end43
  %37 = load i32, i32* %k, align 4
  %inc45 = add nsw i32 %37, 1
  store i32 %inc45, i32* %k, align 4
  br label %for.cond7, !llvm.loop !5

for.end46:                                        ; preds = %for.cond7
  br label %for.inc47

for.inc47:                                        ; preds = %for.end46
  %38 = load i32, i32* %i, align 4
  %inc48 = add nsw i32 %38, 1
  store i32 %inc48, i32* %i, align 4
  br label %for.cond, !llvm.loop !6

for.end49:                                        ; preds = %for.cond
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
!6 = distinct !{!6, !3}
