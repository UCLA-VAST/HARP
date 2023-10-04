; ModuleID = 'trmm.c'
source_filename = "trmm.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @kernel_trmm(double %alpha, [60 x double]* %A, [80 x double]* %B) #0 {
entry:
  %alpha.addr = alloca double, align 8
  %A.addr = alloca [60 x double]*, align 8
  %B.addr = alloca [80 x double]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store double %alpha, double* %alpha.addr, align 8
  store [60 x double]* %A, [60 x double]** %A.addr, align 8
  store [80 x double]* %B, [80 x double]** %B.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc30, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 60
  br i1 %cmp, label %for.body, label %for.end32

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc27, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 80
  br i1 %cmp2, label %for.body3, label %for.end29

for.body3:                                        ; preds = %for.cond1
  store i32 0, i32* %k, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %2 = load i32, i32* %k, align 4
  %cmp5 = icmp slt i32 %2, 60
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %3 = load i32, i32* %k, align 4
  %4 = load i32, i32* %i, align 4
  %cmp7 = icmp sgt i32 %3, %4
  br i1 %cmp7, label %if.then, label %if.end

if.then:                                          ; preds = %for.body6
  %5 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %6 = load i32, i32* %k, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds [60 x double], [60 x double]* %5, i64 %idxprom
  %7 = load i32, i32* %i, align 4
  %idxprom8 = sext i32 %7 to i64
  %arrayidx9 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx, i64 0, i64 %idxprom8
  %8 = load double, double* %arrayidx9, align 8
  %9 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %10 = load i32, i32* %k, align 4
  %idxprom10 = sext i32 %10 to i64
  %arrayidx11 = getelementptr inbounds [80 x double], [80 x double]* %9, i64 %idxprom10
  %11 = load i32, i32* %j, align 4
  %idxprom12 = sext i32 %11 to i64
  %arrayidx13 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx11, i64 0, i64 %idxprom12
  %12 = load double, double* %arrayidx13, align 8
  %mul = fmul double %8, %12
  %13 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %14 = load i32, i32* %i, align 4
  %idxprom14 = sext i32 %14 to i64
  %arrayidx15 = getelementptr inbounds [80 x double], [80 x double]* %13, i64 %idxprom14
  %15 = load i32, i32* %j, align 4
  %idxprom16 = sext i32 %15 to i64
  %arrayidx17 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx15, i64 0, i64 %idxprom16
  %16 = load double, double* %arrayidx17, align 8
  %add = fadd double %16, %mul
  store double %add, double* %arrayidx17, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body6
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %17 = load i32, i32* %k, align 4
  %inc = add nsw i32 %17, 1
  store i32 %inc, i32* %k, align 4
  br label %for.cond4, !llvm.loop !2

for.end:                                          ; preds = %for.cond4
  %18 = load double, double* %alpha.addr, align 8
  %19 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %20 = load i32, i32* %i, align 4
  %idxprom18 = sext i32 %20 to i64
  %arrayidx19 = getelementptr inbounds [80 x double], [80 x double]* %19, i64 %idxprom18
  %21 = load i32, i32* %j, align 4
  %idxprom20 = sext i32 %21 to i64
  %arrayidx21 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx19, i64 0, i64 %idxprom20
  %22 = load double, double* %arrayidx21, align 8
  %mul22 = fmul double %18, %22
  %23 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %24 = load i32, i32* %i, align 4
  %idxprom23 = sext i32 %24 to i64
  %arrayidx24 = getelementptr inbounds [80 x double], [80 x double]* %23, i64 %idxprom23
  %25 = load i32, i32* %j, align 4
  %idxprom25 = sext i32 %25 to i64
  %arrayidx26 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx24, i64 0, i64 %idxprom25
  store double %mul22, double* %arrayidx26, align 8
  br label %for.inc27

for.inc27:                                        ; preds = %for.end
  %26 = load i32, i32* %j, align 4
  %inc28 = add nsw i32 %26, 1
  store i32 %inc28, i32* %j, align 4
  br label %for.cond1, !llvm.loop !4

for.end29:                                        ; preds = %for.cond1
  br label %for.inc30

for.inc30:                                        ; preds = %for.end29
  %27 = load i32, i32* %i, align 4
  %inc31 = add nsw i32 %27, 1
  store i32 %inc31, i32* %i, align 4
  br label %for.cond, !llvm.loop !5

for.end32:                                        ; preds = %for.cond
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
