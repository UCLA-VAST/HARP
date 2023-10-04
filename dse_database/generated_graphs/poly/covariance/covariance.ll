; ModuleID = 'covariance.c'
source_filename = "covariance.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @kernel_covariance(i32 %m, i32 %n, double %float_n, [80 x double]* %data, [80 x double]* %cov, double* %mean) #0 {
entry:
  %m.addr = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %float_n.addr = alloca double, align 8
  %data.addr = alloca [80 x double]*, align 8
  %cov.addr = alloca [80 x double]*, align 8
  %mean.addr = alloca double*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 %m, i32* %m.addr, align 4
  store i32 %n, i32* %n.addr, align 4
  store double %float_n, double* %float_n.addr, align 8
  store [80 x double]* %data, [80 x double]** %data.addr, align 8
  store [80 x double]* %cov, [80 x double]** %cov.addr, align 8
  store double* %mean, double** %mean.addr, align 8
  store i32 0, i32* %j, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc12, %entry
  %0 = load i32, i32* %j, align 4
  %cmp = icmp slt i32 %0, 80
  br i1 %cmp, label %for.body, label %for.end14

for.body:                                         ; preds = %for.cond
  %1 = load double*, double** %mean.addr, align 8
  %2 = load i32, i32* %j, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds double, double* %1, i64 %idxprom
  store double 0.000000e+00, double* %arrayidx, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %3 = load i32, i32* %i, align 4
  %cmp2 = icmp slt i32 %3, 100
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %4 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %5 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %5 to i64
  %arrayidx5 = getelementptr inbounds [80 x double], [80 x double]* %4, i64 %idxprom4
  %6 = load i32, i32* %j, align 4
  %idxprom6 = sext i32 %6 to i64
  %arrayidx7 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx5, i64 0, i64 %idxprom6
  %7 = load double, double* %arrayidx7, align 8
  %8 = load double*, double** %mean.addr, align 8
  %9 = load i32, i32* %j, align 4
  %idxprom8 = sext i32 %9 to i64
  %arrayidx9 = getelementptr inbounds double, double* %8, i64 %idxprom8
  %10 = load double, double* %arrayidx9, align 8
  %add = fadd double %10, %7
  store double %add, double* %arrayidx9, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %11 = load i32, i32* %i, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond1, !llvm.loop !2

for.end:                                          ; preds = %for.cond1
  %12 = load double, double* %float_n.addr, align 8
  %13 = load double*, double** %mean.addr, align 8
  %14 = load i32, i32* %j, align 4
  %idxprom10 = sext i32 %14 to i64
  %arrayidx11 = getelementptr inbounds double, double* %13, i64 %idxprom10
  %15 = load double, double* %arrayidx11, align 8
  %div = fdiv double %15, %12
  store double %div, double* %arrayidx11, align 8
  br label %for.inc12

for.inc12:                                        ; preds = %for.end
  %16 = load i32, i32* %j, align 4
  %inc13 = add nsw i32 %16, 1
  store i32 %inc13, i32* %j, align 4
  br label %for.cond, !llvm.loop !4

for.end14:                                        ; preds = %for.cond
  store i32 0, i32* %i, align 4
  br label %for.cond15

for.cond15:                                       ; preds = %for.inc30, %for.end14
  %17 = load i32, i32* %i, align 4
  %cmp16 = icmp slt i32 %17, 100
  br i1 %cmp16, label %for.body17, label %for.end32

for.body17:                                       ; preds = %for.cond15
  store i32 0, i32* %j, align 4
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc27, %for.body17
  %18 = load i32, i32* %j, align 4
  %cmp19 = icmp slt i32 %18, 80
  br i1 %cmp19, label %for.body20, label %for.end29

for.body20:                                       ; preds = %for.cond18
  %19 = load double*, double** %mean.addr, align 8
  %20 = load i32, i32* %j, align 4
  %idxprom21 = sext i32 %20 to i64
  %arrayidx22 = getelementptr inbounds double, double* %19, i64 %idxprom21
  %21 = load double, double* %arrayidx22, align 8
  %22 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %23 = load i32, i32* %i, align 4
  %idxprom23 = sext i32 %23 to i64
  %arrayidx24 = getelementptr inbounds [80 x double], [80 x double]* %22, i64 %idxprom23
  %24 = load i32, i32* %j, align 4
  %idxprom25 = sext i32 %24 to i64
  %arrayidx26 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx24, i64 0, i64 %idxprom25
  %25 = load double, double* %arrayidx26, align 8
  %sub = fsub double %25, %21
  store double %sub, double* %arrayidx26, align 8
  br label %for.inc27

for.inc27:                                        ; preds = %for.body20
  %26 = load i32, i32* %j, align 4
  %inc28 = add nsw i32 %26, 1
  store i32 %inc28, i32* %j, align 4
  br label %for.cond18, !llvm.loop !5

for.end29:                                        ; preds = %for.cond18
  br label %for.inc30

for.inc30:                                        ; preds = %for.end29
  %27 = load i32, i32* %i, align 4
  %inc31 = add nsw i32 %27, 1
  store i32 %inc31, i32* %i, align 4
  br label %for.cond15, !llvm.loop !6

for.end32:                                        ; preds = %for.cond15
  store i32 0, i32* %i, align 4
  br label %for.cond33

for.cond33:                                       ; preds = %for.inc79, %for.end32
  %28 = load i32, i32* %i, align 4
  %cmp34 = icmp slt i32 %28, 80
  br i1 %cmp34, label %for.body35, label %for.end81

for.body35:                                       ; preds = %for.cond33
  %29 = load i32, i32* %i, align 4
  store i32 %29, i32* %j, align 4
  br label %for.cond36

for.cond36:                                       ; preds = %for.inc76, %for.body35
  %30 = load i32, i32* %j, align 4
  %cmp37 = icmp slt i32 %30, 80
  br i1 %cmp37, label %for.body38, label %for.end78

for.body38:                                       ; preds = %for.cond36
  %31 = load [80 x double]*, [80 x double]** %cov.addr, align 8
  %32 = load i32, i32* %i, align 4
  %idxprom39 = sext i32 %32 to i64
  %arrayidx40 = getelementptr inbounds [80 x double], [80 x double]* %31, i64 %idxprom39
  %33 = load i32, i32* %j, align 4
  %idxprom41 = sext i32 %33 to i64
  %arrayidx42 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx40, i64 0, i64 %idxprom41
  store double 0.000000e+00, double* %arrayidx42, align 8
  store i32 0, i32* %k, align 4
  br label %for.cond43

for.cond43:                                       ; preds = %for.inc59, %for.body38
  %34 = load i32, i32* %k, align 4
  %cmp44 = icmp slt i32 %34, 100
  br i1 %cmp44, label %for.body45, label %for.end61

for.body45:                                       ; preds = %for.cond43
  %35 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %36 = load i32, i32* %k, align 4
  %idxprom46 = sext i32 %36 to i64
  %arrayidx47 = getelementptr inbounds [80 x double], [80 x double]* %35, i64 %idxprom46
  %37 = load i32, i32* %i, align 4
  %idxprom48 = sext i32 %37 to i64
  %arrayidx49 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx47, i64 0, i64 %idxprom48
  %38 = load double, double* %arrayidx49, align 8
  %39 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %40 = load i32, i32* %k, align 4
  %idxprom50 = sext i32 %40 to i64
  %arrayidx51 = getelementptr inbounds [80 x double], [80 x double]* %39, i64 %idxprom50
  %41 = load i32, i32* %j, align 4
  %idxprom52 = sext i32 %41 to i64
  %arrayidx53 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx51, i64 0, i64 %idxprom52
  %42 = load double, double* %arrayidx53, align 8
  %mul = fmul double %38, %42
  %43 = load [80 x double]*, [80 x double]** %cov.addr, align 8
  %44 = load i32, i32* %i, align 4
  %idxprom54 = sext i32 %44 to i64
  %arrayidx55 = getelementptr inbounds [80 x double], [80 x double]* %43, i64 %idxprom54
  %45 = load i32, i32* %j, align 4
  %idxprom56 = sext i32 %45 to i64
  %arrayidx57 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx55, i64 0, i64 %idxprom56
  %46 = load double, double* %arrayidx57, align 8
  %add58 = fadd double %46, %mul
  store double %add58, double* %arrayidx57, align 8
  br label %for.inc59

for.inc59:                                        ; preds = %for.body45
  %47 = load i32, i32* %k, align 4
  %inc60 = add nsw i32 %47, 1
  store i32 %inc60, i32* %k, align 4
  br label %for.cond43, !llvm.loop !7

for.end61:                                        ; preds = %for.cond43
  %48 = load double, double* %float_n.addr, align 8
  %sub62 = fsub double %48, 1.000000e+00
  %49 = load [80 x double]*, [80 x double]** %cov.addr, align 8
  %50 = load i32, i32* %i, align 4
  %idxprom63 = sext i32 %50 to i64
  %arrayidx64 = getelementptr inbounds [80 x double], [80 x double]* %49, i64 %idxprom63
  %51 = load i32, i32* %j, align 4
  %idxprom65 = sext i32 %51 to i64
  %arrayidx66 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx64, i64 0, i64 %idxprom65
  %52 = load double, double* %arrayidx66, align 8
  %div67 = fdiv double %52, %sub62
  store double %div67, double* %arrayidx66, align 8
  %53 = load [80 x double]*, [80 x double]** %cov.addr, align 8
  %54 = load i32, i32* %i, align 4
  %idxprom68 = sext i32 %54 to i64
  %arrayidx69 = getelementptr inbounds [80 x double], [80 x double]* %53, i64 %idxprom68
  %55 = load i32, i32* %j, align 4
  %idxprom70 = sext i32 %55 to i64
  %arrayidx71 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx69, i64 0, i64 %idxprom70
  %56 = load double, double* %arrayidx71, align 8
  %57 = load [80 x double]*, [80 x double]** %cov.addr, align 8
  %58 = load i32, i32* %j, align 4
  %idxprom72 = sext i32 %58 to i64
  %arrayidx73 = getelementptr inbounds [80 x double], [80 x double]* %57, i64 %idxprom72
  %59 = load i32, i32* %i, align 4
  %idxprom74 = sext i32 %59 to i64
  %arrayidx75 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx73, i64 0, i64 %idxprom74
  store double %56, double* %arrayidx75, align 8
  br label %for.inc76

for.inc76:                                        ; preds = %for.end61
  %60 = load i32, i32* %j, align 4
  %inc77 = add nsw i32 %60, 1
  store i32 %inc77, i32* %j, align 4
  br label %for.cond36, !llvm.loop !8

for.end78:                                        ; preds = %for.cond36
  br label %for.inc79

for.inc79:                                        ; preds = %for.end78
  %61 = load i32, i32* %i, align 4
  %inc80 = add nsw i32 %61, 1
  store i32 %inc80, i32* %i, align 4
  br label %for.cond33, !llvm.loop !9

for.end81:                                        ; preds = %for.cond33
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
!8 = distinct !{!8, !3}
!9 = distinct !{!9, !3}
