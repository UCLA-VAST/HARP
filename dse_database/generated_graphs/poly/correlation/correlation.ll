; ModuleID = 'correlation.c'
source_filename = "correlation.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @kernel_correlation(double %float_n, [80 x double]* %data, [80 x double]* %corr, double* %mean, double* %stddev) #0 {
entry:
  %float_n.addr = alloca double, align 8
  %data.addr = alloca [80 x double]*, align 8
  %corr.addr = alloca [80 x double]*, align 8
  %mean.addr = alloca double*, align 8
  %stddev.addr = alloca double*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %eps = alloca double, align 8
  store double %float_n, double* %float_n.addr, align 8
  store [80 x double]* %data, [80 x double]** %data.addr, align 8
  store [80 x double]* %corr, [80 x double]** %corr.addr, align 8
  store double* %mean, double** %mean.addr, align 8
  store double* %stddev, double** %stddev.addr, align 8
  store double 1.000000e-01, double* %eps, align 8
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
  store i32 0, i32* %j, align 4
  br label %for.cond15

for.cond15:                                       ; preds = %for.inc50, %for.end14
  %17 = load i32, i32* %j, align 4
  %cmp16 = icmp slt i32 %17, 80
  br i1 %cmp16, label %for.body17, label %for.end52

for.body17:                                       ; preds = %for.cond15
  %18 = load double*, double** %stddev.addr, align 8
  %19 = load i32, i32* %j, align 4
  %idxprom18 = sext i32 %19 to i64
  %arrayidx19 = getelementptr inbounds double, double* %18, i64 %idxprom18
  store double 0.000000e+00, double* %arrayidx19, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond20

for.cond20:                                       ; preds = %for.inc32, %for.body17
  %20 = load i32, i32* %i, align 4
  %cmp21 = icmp slt i32 %20, 100
  br i1 %cmp21, label %for.body22, label %for.end34

for.body22:                                       ; preds = %for.cond20
  %21 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %22 = load i32, i32* %i, align 4
  %idxprom23 = sext i32 %22 to i64
  %arrayidx24 = getelementptr inbounds [80 x double], [80 x double]* %21, i64 %idxprom23
  %23 = load i32, i32* %j, align 4
  %idxprom25 = sext i32 %23 to i64
  %arrayidx26 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx24, i64 0, i64 %idxprom25
  %24 = load double, double* %arrayidx26, align 8
  %25 = load double*, double** %mean.addr, align 8
  %26 = load i32, i32* %j, align 4
  %idxprom27 = sext i32 %26 to i64
  %arrayidx28 = getelementptr inbounds double, double* %25, i64 %idxprom27
  %27 = load double, double* %arrayidx28, align 8
  %sub = fsub double %24, %27
  %call = call double @pow(double %sub, double 2.000000e+00) #2
  %28 = load double*, double** %stddev.addr, align 8
  %29 = load i32, i32* %j, align 4
  %idxprom29 = sext i32 %29 to i64
  %arrayidx30 = getelementptr inbounds double, double* %28, i64 %idxprom29
  %30 = load double, double* %arrayidx30, align 8
  %add31 = fadd double %30, %call
  store double %add31, double* %arrayidx30, align 8
  br label %for.inc32

for.inc32:                                        ; preds = %for.body22
  %31 = load i32, i32* %i, align 4
  %inc33 = add nsw i32 %31, 1
  store i32 %inc33, i32* %i, align 4
  br label %for.cond20, !llvm.loop !5

for.end34:                                        ; preds = %for.cond20
  %32 = load double, double* %float_n.addr, align 8
  %33 = load double*, double** %stddev.addr, align 8
  %34 = load i32, i32* %j, align 4
  %idxprom35 = sext i32 %34 to i64
  %arrayidx36 = getelementptr inbounds double, double* %33, i64 %idxprom35
  %35 = load double, double* %arrayidx36, align 8
  %div37 = fdiv double %35, %32
  store double %div37, double* %arrayidx36, align 8
  %36 = load double*, double** %stddev.addr, align 8
  %37 = load i32, i32* %j, align 4
  %idxprom38 = sext i32 %37 to i64
  %arrayidx39 = getelementptr inbounds double, double* %36, i64 %idxprom38
  %38 = load double, double* %arrayidx39, align 8
  %call40 = call double @sqrt(double %38) #2
  %39 = load double*, double** %stddev.addr, align 8
  %40 = load i32, i32* %j, align 4
  %idxprom41 = sext i32 %40 to i64
  %arrayidx42 = getelementptr inbounds double, double* %39, i64 %idxprom41
  store double %call40, double* %arrayidx42, align 8
  %41 = load double*, double** %stddev.addr, align 8
  %42 = load i32, i32* %j, align 4
  %idxprom43 = sext i32 %42 to i64
  %arrayidx44 = getelementptr inbounds double, double* %41, i64 %idxprom43
  %43 = load double, double* %arrayidx44, align 8
  %44 = load double, double* %eps, align 8
  %cmp45 = fcmp ole double %43, %44
  br i1 %cmp45, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.end34
  br label %cond.end

cond.false:                                       ; preds = %for.end34
  %45 = load double*, double** %stddev.addr, align 8
  %46 = load i32, i32* %j, align 4
  %idxprom46 = sext i32 %46 to i64
  %arrayidx47 = getelementptr inbounds double, double* %45, i64 %idxprom46
  %47 = load double, double* %arrayidx47, align 8
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ 1.000000e+00, %cond.true ], [ %47, %cond.false ]
  %48 = load double*, double** %stddev.addr, align 8
  %49 = load i32, i32* %j, align 4
  %idxprom48 = sext i32 %49 to i64
  %arrayidx49 = getelementptr inbounds double, double* %48, i64 %idxprom48
  store double %cond, double* %arrayidx49, align 8
  br label %for.inc50

for.inc50:                                        ; preds = %cond.end
  %50 = load i32, i32* %j, align 4
  %inc51 = add nsw i32 %50, 1
  store i32 %inc51, i32* %j, align 4
  br label %for.cond15, !llvm.loop !6

for.end52:                                        ; preds = %for.cond15
  store i32 0, i32* %i, align 4
  br label %for.cond53

for.cond53:                                       ; preds = %for.inc77, %for.end52
  %51 = load i32, i32* %i, align 4
  %cmp54 = icmp slt i32 %51, 100
  br i1 %cmp54, label %for.body55, label %for.end79

for.body55:                                       ; preds = %for.cond53
  store i32 0, i32* %j, align 4
  br label %for.cond56

for.cond56:                                       ; preds = %for.inc74, %for.body55
  %52 = load i32, i32* %j, align 4
  %cmp57 = icmp slt i32 %52, 80
  br i1 %cmp57, label %for.body58, label %for.end76

for.body58:                                       ; preds = %for.cond56
  %53 = load double*, double** %mean.addr, align 8
  %54 = load i32, i32* %j, align 4
  %idxprom59 = sext i32 %54 to i64
  %arrayidx60 = getelementptr inbounds double, double* %53, i64 %idxprom59
  %55 = load double, double* %arrayidx60, align 8
  %56 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %57 = load i32, i32* %i, align 4
  %idxprom61 = sext i32 %57 to i64
  %arrayidx62 = getelementptr inbounds [80 x double], [80 x double]* %56, i64 %idxprom61
  %58 = load i32, i32* %j, align 4
  %idxprom63 = sext i32 %58 to i64
  %arrayidx64 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx62, i64 0, i64 %idxprom63
  %59 = load double, double* %arrayidx64, align 8
  %sub65 = fsub double %59, %55
  store double %sub65, double* %arrayidx64, align 8
  %60 = load double, double* %float_n.addr, align 8
  %call66 = call double @sqrt(double %60) #2
  %61 = load double*, double** %stddev.addr, align 8
  %62 = load i32, i32* %j, align 4
  %idxprom67 = sext i32 %62 to i64
  %arrayidx68 = getelementptr inbounds double, double* %61, i64 %idxprom67
  %63 = load double, double* %arrayidx68, align 8
  %mul = fmul double %call66, %63
  %64 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %65 = load i32, i32* %i, align 4
  %idxprom69 = sext i32 %65 to i64
  %arrayidx70 = getelementptr inbounds [80 x double], [80 x double]* %64, i64 %idxprom69
  %66 = load i32, i32* %j, align 4
  %idxprom71 = sext i32 %66 to i64
  %arrayidx72 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx70, i64 0, i64 %idxprom71
  %67 = load double, double* %arrayidx72, align 8
  %div73 = fdiv double %67, %mul
  store double %div73, double* %arrayidx72, align 8
  br label %for.inc74

for.inc74:                                        ; preds = %for.body58
  %68 = load i32, i32* %j, align 4
  %inc75 = add nsw i32 %68, 1
  store i32 %inc75, i32* %j, align 4
  br label %for.cond56, !llvm.loop !7

for.end76:                                        ; preds = %for.cond56
  br label %for.inc77

for.inc77:                                        ; preds = %for.end76
  %69 = load i32, i32* %i, align 4
  %inc78 = add nsw i32 %69, 1
  store i32 %inc78, i32* %i, align 4
  br label %for.cond53, !llvm.loop !8

for.end79:                                        ; preds = %for.cond53
  store i32 0, i32* %i, align 4
  br label %for.cond80

for.cond80:                                       ; preds = %for.inc126, %for.end79
  %70 = load i32, i32* %i, align 4
  %cmp81 = icmp slt i32 %70, 79
  br i1 %cmp81, label %for.body82, label %for.end128

for.body82:                                       ; preds = %for.cond80
  %71 = load [80 x double]*, [80 x double]** %corr.addr, align 8
  %72 = load i32, i32* %i, align 4
  %idxprom83 = sext i32 %72 to i64
  %arrayidx84 = getelementptr inbounds [80 x double], [80 x double]* %71, i64 %idxprom83
  %73 = load i32, i32* %i, align 4
  %idxprom85 = sext i32 %73 to i64
  %arrayidx86 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx84, i64 0, i64 %idxprom85
  store double 1.000000e+00, double* %arrayidx86, align 8
  %74 = load i32, i32* %i, align 4
  %add87 = add nsw i32 %74, 1
  store i32 %add87, i32* %j, align 4
  br label %for.cond88

for.cond88:                                       ; preds = %for.inc123, %for.body82
  %75 = load i32, i32* %j, align 4
  %cmp89 = icmp slt i32 %75, 80
  br i1 %cmp89, label %for.body90, label %for.end125

for.body90:                                       ; preds = %for.cond88
  %76 = load [80 x double]*, [80 x double]** %corr.addr, align 8
  %77 = load i32, i32* %i, align 4
  %idxprom91 = sext i32 %77 to i64
  %arrayidx92 = getelementptr inbounds [80 x double], [80 x double]* %76, i64 %idxprom91
  %78 = load i32, i32* %j, align 4
  %idxprom93 = sext i32 %78 to i64
  %arrayidx94 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx92, i64 0, i64 %idxprom93
  store double 0.000000e+00, double* %arrayidx94, align 8
  store i32 0, i32* %k, align 4
  br label %for.cond95

for.cond95:                                       ; preds = %for.inc112, %for.body90
  %79 = load i32, i32* %k, align 4
  %cmp96 = icmp slt i32 %79, 100
  br i1 %cmp96, label %for.body97, label %for.end114

for.body97:                                       ; preds = %for.cond95
  %80 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %81 = load i32, i32* %k, align 4
  %idxprom98 = sext i32 %81 to i64
  %arrayidx99 = getelementptr inbounds [80 x double], [80 x double]* %80, i64 %idxprom98
  %82 = load i32, i32* %i, align 4
  %idxprom100 = sext i32 %82 to i64
  %arrayidx101 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx99, i64 0, i64 %idxprom100
  %83 = load double, double* %arrayidx101, align 8
  %84 = load [80 x double]*, [80 x double]** %data.addr, align 8
  %85 = load i32, i32* %k, align 4
  %idxprom102 = sext i32 %85 to i64
  %arrayidx103 = getelementptr inbounds [80 x double], [80 x double]* %84, i64 %idxprom102
  %86 = load i32, i32* %j, align 4
  %idxprom104 = sext i32 %86 to i64
  %arrayidx105 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx103, i64 0, i64 %idxprom104
  %87 = load double, double* %arrayidx105, align 8
  %mul106 = fmul double %83, %87
  %88 = load [80 x double]*, [80 x double]** %corr.addr, align 8
  %89 = load i32, i32* %i, align 4
  %idxprom107 = sext i32 %89 to i64
  %arrayidx108 = getelementptr inbounds [80 x double], [80 x double]* %88, i64 %idxprom107
  %90 = load i32, i32* %j, align 4
  %idxprom109 = sext i32 %90 to i64
  %arrayidx110 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx108, i64 0, i64 %idxprom109
  %91 = load double, double* %arrayidx110, align 8
  %add111 = fadd double %91, %mul106
  store double %add111, double* %arrayidx110, align 8
  br label %for.inc112

for.inc112:                                       ; preds = %for.body97
  %92 = load i32, i32* %k, align 4
  %inc113 = add nsw i32 %92, 1
  store i32 %inc113, i32* %k, align 4
  br label %for.cond95, !llvm.loop !9

for.end114:                                       ; preds = %for.cond95
  %93 = load [80 x double]*, [80 x double]** %corr.addr, align 8
  %94 = load i32, i32* %i, align 4
  %idxprom115 = sext i32 %94 to i64
  %arrayidx116 = getelementptr inbounds [80 x double], [80 x double]* %93, i64 %idxprom115
  %95 = load i32, i32* %j, align 4
  %idxprom117 = sext i32 %95 to i64
  %arrayidx118 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx116, i64 0, i64 %idxprom117
  %96 = load double, double* %arrayidx118, align 8
  %97 = load [80 x double]*, [80 x double]** %corr.addr, align 8
  %98 = load i32, i32* %j, align 4
  %idxprom119 = sext i32 %98 to i64
  %arrayidx120 = getelementptr inbounds [80 x double], [80 x double]* %97, i64 %idxprom119
  %99 = load i32, i32* %i, align 4
  %idxprom121 = sext i32 %99 to i64
  %arrayidx122 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx120, i64 0, i64 %idxprom121
  store double %96, double* %arrayidx122, align 8
  br label %for.inc123

for.inc123:                                       ; preds = %for.end114
  %100 = load i32, i32* %j, align 4
  %inc124 = add nsw i32 %100, 1
  store i32 %inc124, i32* %j, align 4
  br label %for.cond88, !llvm.loop !10

for.end125:                                       ; preds = %for.cond88
  br label %for.inc126

for.inc126:                                       ; preds = %for.end125
  %101 = load i32, i32* %i, align 4
  %inc127 = add nsw i32 %101, 1
  store i32 %inc127, i32* %i, align 4
  br label %for.cond80, !llvm.loop !11

for.end128:                                       ; preds = %for.cond80
  %102 = load [80 x double]*, [80 x double]** %corr.addr, align 8
  %arrayidx129 = getelementptr inbounds [80 x double], [80 x double]* %102, i64 79
  %arrayidx130 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx129, i64 0, i64 79
  store double 1.000000e+00, double* %arrayidx130, align 8
  ret void
}

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #1

; Function Attrs: nounwind
declare dso_local double @sqrt(double) #1

attributes #0 = { noinline nounwind optnone uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

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
!10 = distinct !{!10, !3}
!11 = distinct !{!11, !3}
