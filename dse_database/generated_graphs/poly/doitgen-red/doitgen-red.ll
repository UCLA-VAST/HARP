; ModuleID = 'doitgen-red.c'
source_filename = "doitgen-red.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @kernel_doitgen([20 x [30 x double]]* %A, [30 x double]* %C4, double* %sum) #0 {
entry:
  %A.addr = alloca [20 x [30 x double]]*, align 8
  %C4.addr = alloca [30 x double]*, align 8
  %sum.addr = alloca double*, align 8
  %r = alloca i32, align 4
  %q = alloca i32, align 4
  %p = alloca i32, align 4
  %s = alloca i32, align 4
  %sum_tmp = alloca double, align 8
  store [20 x [30 x double]]* %A, [20 x [30 x double]]** %A.addr, align 8
  store [30 x double]* %C4, [30 x double]** %C4.addr, align 8
  store double* %sum, double** %sum.addr, align 8
  store i32 0, i32* %r, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc40, %entry
  %0 = load i32, i32* %r, align 4
  %cmp = icmp slt i32 %0, 25
  br i1 %cmp, label %for.body, label %for.end42

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %q, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc37, %for.body
  %1 = load i32, i32* %q, align 4
  %cmp2 = icmp slt i32 %1, 20
  br i1 %cmp2, label %for.body3, label %for.end39

for.body3:                                        ; preds = %for.cond1
  store i32 0, i32* %p, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc20, %for.body3
  %2 = load i32, i32* %p, align 4
  %cmp5 = icmp slt i32 %2, 30
  br i1 %cmp5, label %for.body6, label %for.end22

for.body6:                                        ; preds = %for.cond4
  store double 0.000000e+00, double* %sum_tmp, align 8
  store i32 0, i32* %s, align 4
  br label %for.cond7

for.cond7:                                        ; preds = %for.inc, %for.body6
  %3 = load i32, i32* %s, align 4
  %cmp8 = icmp slt i32 %3, 30
  br i1 %cmp8, label %for.body9, label %for.end

for.body9:                                        ; preds = %for.cond7
  %4 = load [20 x [30 x double]]*, [20 x [30 x double]]** %A.addr, align 8
  %5 = load i32, i32* %r, align 4
  %idxprom = sext i32 %5 to i64
  %arrayidx = getelementptr inbounds [20 x [30 x double]], [20 x [30 x double]]* %4, i64 %idxprom
  %6 = load i32, i32* %q, align 4
  %idxprom10 = sext i32 %6 to i64
  %arrayidx11 = getelementptr inbounds [20 x [30 x double]], [20 x [30 x double]]* %arrayidx, i64 0, i64 %idxprom10
  %7 = load i32, i32* %s, align 4
  %idxprom12 = sext i32 %7 to i64
  %arrayidx13 = getelementptr inbounds [30 x double], [30 x double]* %arrayidx11, i64 0, i64 %idxprom12
  %8 = load double, double* %arrayidx13, align 8
  %9 = load [30 x double]*, [30 x double]** %C4.addr, align 8
  %10 = load i32, i32* %s, align 4
  %idxprom14 = sext i32 %10 to i64
  %arrayidx15 = getelementptr inbounds [30 x double], [30 x double]* %9, i64 %idxprom14
  %11 = load i32, i32* %p, align 4
  %idxprom16 = sext i32 %11 to i64
  %arrayidx17 = getelementptr inbounds [30 x double], [30 x double]* %arrayidx15, i64 0, i64 %idxprom16
  %12 = load double, double* %arrayidx17, align 8
  %mul = fmul double %8, %12
  %13 = load double, double* %sum_tmp, align 8
  %add = fadd double %13, %mul
  store double %add, double* %sum_tmp, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body9
  %14 = load i32, i32* %s, align 4
  %inc = add nsw i32 %14, 1
  store i32 %inc, i32* %s, align 4
  br label %for.cond7, !llvm.loop !2

for.end:                                          ; preds = %for.cond7
  %15 = load double, double* %sum_tmp, align 8
  %16 = load double*, double** %sum.addr, align 8
  %17 = load i32, i32* %p, align 4
  %idxprom18 = sext i32 %17 to i64
  %arrayidx19 = getelementptr inbounds double, double* %16, i64 %idxprom18
  store double %15, double* %arrayidx19, align 8
  br label %for.inc20

for.inc20:                                        ; preds = %for.end
  %18 = load i32, i32* %p, align 4
  %inc21 = add nsw i32 %18, 1
  store i32 %inc21, i32* %p, align 4
  br label %for.cond4, !llvm.loop !4

for.end22:                                        ; preds = %for.cond4
  store i32 0, i32* %p, align 4
  br label %for.cond23

for.cond23:                                       ; preds = %for.inc34, %for.end22
  %19 = load i32, i32* %p, align 4
  %cmp24 = icmp slt i32 %19, 30
  br i1 %cmp24, label %for.body25, label %for.end36

for.body25:                                       ; preds = %for.cond23
  %20 = load double*, double** %sum.addr, align 8
  %21 = load i32, i32* %p, align 4
  %idxprom26 = sext i32 %21 to i64
  %arrayidx27 = getelementptr inbounds double, double* %20, i64 %idxprom26
  %22 = load double, double* %arrayidx27, align 8
  %23 = load [20 x [30 x double]]*, [20 x [30 x double]]** %A.addr, align 8
  %24 = load i32, i32* %r, align 4
  %idxprom28 = sext i32 %24 to i64
  %arrayidx29 = getelementptr inbounds [20 x [30 x double]], [20 x [30 x double]]* %23, i64 %idxprom28
  %25 = load i32, i32* %q, align 4
  %idxprom30 = sext i32 %25 to i64
  %arrayidx31 = getelementptr inbounds [20 x [30 x double]], [20 x [30 x double]]* %arrayidx29, i64 0, i64 %idxprom30
  %26 = load i32, i32* %p, align 4
  %idxprom32 = sext i32 %26 to i64
  %arrayidx33 = getelementptr inbounds [30 x double], [30 x double]* %arrayidx31, i64 0, i64 %idxprom32
  store double %22, double* %arrayidx33, align 8
  br label %for.inc34

for.inc34:                                        ; preds = %for.body25
  %27 = load i32, i32* %p, align 4
  %inc35 = add nsw i32 %27, 1
  store i32 %inc35, i32* %p, align 4
  br label %for.cond23, !llvm.loop !5

for.end36:                                        ; preds = %for.cond23
  br label %for.inc37

for.inc37:                                        ; preds = %for.end36
  %28 = load i32, i32* %q, align 4
  %inc38 = add nsw i32 %28, 1
  store i32 %inc38, i32* %q, align 4
  br label %for.cond1, !llvm.loop !6

for.end39:                                        ; preds = %for.cond1
  br label %for.inc40

for.inc40:                                        ; preds = %for.end39
  %29 = load i32, i32* %r, align 4
  %inc41 = add nsw i32 %29, 1
  store i32 %inc41, i32* %r, align 4
  br label %for.cond, !llvm.loop !7

for.end42:                                        ; preds = %for.cond
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
!7 = distinct !{!7, !3}
