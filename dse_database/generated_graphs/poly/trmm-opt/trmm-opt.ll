; ModuleID = 'trmm-opt.c'
source_filename = "trmm-opt.c"
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
  %sum = alloca double, align 8
  %k = alloca i32, align 4
  store double %alpha, double* %alpha.addr, align 8
  store [60 x double]* %A, [60 x double]** %A.addr, align 8
  store [80 x double]* %B, [80 x double]** %B.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc26, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 60
  br i1 %cmp, label %for.body, label %for.end28

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc23, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 80
  br i1 %cmp2, label %for.body3, label %for.end25

for.body3:                                        ; preds = %for.cond1
  %2 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [80 x double], [80 x double]* %2, i64 %idxprom
  %4 = load i32, i32* %j, align 4
  %idxprom4 = sext i32 %4 to i64
  %arrayidx5 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx, i64 0, i64 %idxprom4
  %5 = load double, double* %arrayidx5, align 8
  store double %5, double* %sum, align 8
  store i32 0, i32* %k, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body3
  %6 = load i32, i32* %k, align 4
  %cmp7 = icmp slt i32 %6, 60
  br i1 %cmp7, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %7 = load i32, i32* %k, align 4
  %8 = load i32, i32* %i, align 4
  %cmp9 = icmp sgt i32 %7, %8
  br i1 %cmp9, label %if.then, label %if.end

if.then:                                          ; preds = %for.body8
  %9 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %10 = load i32, i32* %k, align 4
  %idxprom10 = sext i32 %10 to i64
  %arrayidx11 = getelementptr inbounds [60 x double], [60 x double]* %9, i64 %idxprom10
  %11 = load i32, i32* %i, align 4
  %idxprom12 = sext i32 %11 to i64
  %arrayidx13 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx11, i64 0, i64 %idxprom12
  %12 = load double, double* %arrayidx13, align 8
  %13 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %14 = load i32, i32* %k, align 4
  %idxprom14 = sext i32 %14 to i64
  %arrayidx15 = getelementptr inbounds [80 x double], [80 x double]* %13, i64 %idxprom14
  %15 = load i32, i32* %j, align 4
  %idxprom16 = sext i32 %15 to i64
  %arrayidx17 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx15, i64 0, i64 %idxprom16
  %16 = load double, double* %arrayidx17, align 8
  %mul = fmul double %12, %16
  %17 = load double, double* %sum, align 8
  %add = fadd double %17, %mul
  store double %add, double* %sum, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body8
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %18 = load i32, i32* %k, align 4
  %inc = add nsw i32 %18, 1
  store i32 %inc, i32* %k, align 4
  br label %for.cond6, !llvm.loop !2

for.end:                                          ; preds = %for.cond6
  %19 = load double, double* %alpha.addr, align 8
  %20 = load double, double* %sum, align 8
  %mul18 = fmul double %19, %20
  %21 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %22 = load i32, i32* %i, align 4
  %idxprom19 = sext i32 %22 to i64
  %arrayidx20 = getelementptr inbounds [80 x double], [80 x double]* %21, i64 %idxprom19
  %23 = load i32, i32* %j, align 4
  %idxprom21 = sext i32 %23 to i64
  %arrayidx22 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx20, i64 0, i64 %idxprom21
  store double %mul18, double* %arrayidx22, align 8
  br label %for.inc23

for.inc23:                                        ; preds = %for.end
  %24 = load i32, i32* %j, align 4
  %inc24 = add nsw i32 %24, 1
  store i32 %inc24, i32* %j, align 4
  br label %for.cond1, !llvm.loop !4

for.end25:                                        ; preds = %for.cond1
  br label %for.inc26

for.inc26:                                        ; preds = %for.end25
  %25 = load i32, i32* %i, align 4
  %inc27 = add nsw i32 %25, 1
  store i32 %inc27, i32* %i, align 4
  br label %for.cond, !llvm.loop !5

for.end28:                                        ; preds = %for.cond
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
