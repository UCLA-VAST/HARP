; ModuleID = 'gesummv-medium.c'
source_filename = "gesummv-medium.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @kernel_gesummv(double %alpha, double %beta, [250 x double]* %A, [250 x double]* %B, double* %tmp, double* %x, double* %y) #0 {
entry:
  %alpha.addr = alloca double, align 8
  %beta.addr = alloca double, align 8
  %A.addr = alloca [250 x double]*, align 8
  %B.addr = alloca [250 x double]*, align 8
  %tmp.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %y.addr = alloca double*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store double %alpha, double* %alpha.addr, align 8
  store double %beta, double* %beta.addr, align 8
  store [250 x double]* %A, [250 x double]** %A.addr, align 8
  store [250 x double]* %B, [250 x double]** %B.addr, align 8
  store double* %tmp, double** %tmp.addr, align 8
  store double* %x, double** %x.addr, align 8
  store double* %y, double** %y.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc33, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 250
  br i1 %cmp, label %for.body, label %for.end35

for.body:                                         ; preds = %for.cond
  %1 = load double*, double** %tmp.addr, align 8
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds double, double* %1, i64 %idxprom
  store double 0.000000e+00, double* %arrayidx, align 8
  %3 = load double*, double** %y.addr, align 8
  %4 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %4 to i64
  %arrayidx2 = getelementptr inbounds double, double* %3, i64 %idxprom1
  store double 0.000000e+00, double* %arrayidx2, align 8
  store i32 0, i32* %j, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc, %for.body
  %5 = load i32, i32* %j, align 4
  %cmp4 = icmp slt i32 %5, 250
  br i1 %cmp4, label %for.body5, label %for.end

for.body5:                                        ; preds = %for.cond3
  %6 = load [250 x double]*, [250 x double]** %A.addr, align 8
  %7 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %7 to i64
  %arrayidx7 = getelementptr inbounds [250 x double], [250 x double]* %6, i64 %idxprom6
  %8 = load i32, i32* %j, align 4
  %idxprom8 = sext i32 %8 to i64
  %arrayidx9 = getelementptr inbounds [250 x double], [250 x double]* %arrayidx7, i64 0, i64 %idxprom8
  %9 = load double, double* %arrayidx9, align 8
  %10 = load double*, double** %x.addr, align 8
  %11 = load i32, i32* %j, align 4
  %idxprom10 = sext i32 %11 to i64
  %arrayidx11 = getelementptr inbounds double, double* %10, i64 %idxprom10
  %12 = load double, double* %arrayidx11, align 8
  %mul = fmul double %9, %12
  %13 = load double*, double** %tmp.addr, align 8
  %14 = load i32, i32* %i, align 4
  %idxprom12 = sext i32 %14 to i64
  %arrayidx13 = getelementptr inbounds double, double* %13, i64 %idxprom12
  %15 = load double, double* %arrayidx13, align 8
  %add = fadd double %15, %mul
  store double %add, double* %arrayidx13, align 8
  %16 = load [250 x double]*, [250 x double]** %B.addr, align 8
  %17 = load i32, i32* %i, align 4
  %idxprom14 = sext i32 %17 to i64
  %arrayidx15 = getelementptr inbounds [250 x double], [250 x double]* %16, i64 %idxprom14
  %18 = load i32, i32* %j, align 4
  %idxprom16 = sext i32 %18 to i64
  %arrayidx17 = getelementptr inbounds [250 x double], [250 x double]* %arrayidx15, i64 0, i64 %idxprom16
  %19 = load double, double* %arrayidx17, align 8
  %20 = load double*, double** %x.addr, align 8
  %21 = load i32, i32* %j, align 4
  %idxprom18 = sext i32 %21 to i64
  %arrayidx19 = getelementptr inbounds double, double* %20, i64 %idxprom18
  %22 = load double, double* %arrayidx19, align 8
  %mul20 = fmul double %19, %22
  %23 = load double*, double** %y.addr, align 8
  %24 = load i32, i32* %i, align 4
  %idxprom21 = sext i32 %24 to i64
  %arrayidx22 = getelementptr inbounds double, double* %23, i64 %idxprom21
  %25 = load double, double* %arrayidx22, align 8
  %add23 = fadd double %25, %mul20
  store double %add23, double* %arrayidx22, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body5
  %26 = load i32, i32* %j, align 4
  %inc = add nsw i32 %26, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond3, !llvm.loop !2

for.end:                                          ; preds = %for.cond3
  %27 = load double, double* %alpha.addr, align 8
  %28 = load double*, double** %tmp.addr, align 8
  %29 = load i32, i32* %i, align 4
  %idxprom24 = sext i32 %29 to i64
  %arrayidx25 = getelementptr inbounds double, double* %28, i64 %idxprom24
  %30 = load double, double* %arrayidx25, align 8
  %mul26 = fmul double %27, %30
  %31 = load double, double* %beta.addr, align 8
  %32 = load double*, double** %y.addr, align 8
  %33 = load i32, i32* %i, align 4
  %idxprom27 = sext i32 %33 to i64
  %arrayidx28 = getelementptr inbounds double, double* %32, i64 %idxprom27
  %34 = load double, double* %arrayidx28, align 8
  %mul29 = fmul double %31, %34
  %add30 = fadd double %mul26, %mul29
  %35 = load double*, double** %y.addr, align 8
  %36 = load i32, i32* %i, align 4
  %idxprom31 = sext i32 %36 to i64
  %arrayidx32 = getelementptr inbounds double, double* %35, i64 %idxprom31
  store double %add30, double* %arrayidx32, align 8
  br label %for.inc33

for.inc33:                                        ; preds = %for.end
  %37 = load i32, i32* %i, align 4
  %inc34 = add nsw i32 %37, 1
  store i32 %inc34, i32* %i, align 4
  br label %for.cond, !llvm.loop !4

for.end35:                                        ; preds = %for.cond
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
