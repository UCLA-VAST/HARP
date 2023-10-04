; ModuleID = 'symm-opt.c'
source_filename = "symm-opt.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @kernel_symm(double %alpha, double %beta, [80 x double]* %C, [60 x double]* %A, [80 x double]* %B) #0 {
entry:
  %alpha.addr = alloca double, align 8
  %beta.addr = alloca double, align 8
  %C.addr = alloca [80 x double]*, align 8
  %A.addr = alloca [60 x double]*, align 8
  %B.addr = alloca [80 x double]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %tmp = alloca double, align 8
  %temp2 = alloca double, align 8
  store double %alpha, double* %alpha.addr, align 8
  store double %beta, double* %beta.addr, align 8
  store [80 x double]* %C, [80 x double]** %C.addr, align 8
  store [60 x double]* %A, [60 x double]** %A.addr, align 8
  store [80 x double]* %B, [80 x double]** %B.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc63, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 60
  br i1 %cmp, label %for.body, label %for.end65

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc60, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 80
  br i1 %cmp2, label %for.body3, label %for.end62

for.body3:                                        ; preds = %for.cond1
  %2 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [80 x double], [80 x double]* %2, i64 %idxprom
  %4 = load i32, i32* %j, align 4
  %idxprom4 = sext i32 %4 to i64
  %arrayidx5 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx, i64 0, i64 %idxprom4
  %5 = load double, double* %arrayidx5, align 8
  store double %5, double* %tmp, align 8
  store i32 0, i32* %k, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body3
  %6 = load i32, i32* %k, align 4
  %cmp7 = icmp slt i32 %6, 60
  br i1 %cmp7, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %7 = load i32, i32* %k, align 4
  %8 = load i32, i32* %i, align 4
  %cmp9 = icmp slt i32 %7, %8
  br i1 %cmp9, label %if.then, label %if.end

if.then:                                          ; preds = %for.body8
  %9 = load double, double* %alpha.addr, align 8
  %10 = load double, double* %tmp, align 8
  %mul = fmul double %9, %10
  %11 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %12 = load i32, i32* %i, align 4
  %idxprom10 = sext i32 %12 to i64
  %arrayidx11 = getelementptr inbounds [60 x double], [60 x double]* %11, i64 %idxprom10
  %13 = load i32, i32* %k, align 4
  %idxprom12 = sext i32 %13 to i64
  %arrayidx13 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx11, i64 0, i64 %idxprom12
  %14 = load double, double* %arrayidx13, align 8
  %mul14 = fmul double %mul, %14
  %15 = load [80 x double]*, [80 x double]** %C.addr, align 8
  %16 = load i32, i32* %k, align 4
  %idxprom15 = sext i32 %16 to i64
  %arrayidx16 = getelementptr inbounds [80 x double], [80 x double]* %15, i64 %idxprom15
  %17 = load i32, i32* %j, align 4
  %idxprom17 = sext i32 %17 to i64
  %arrayidx18 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx16, i64 0, i64 %idxprom17
  %18 = load double, double* %arrayidx18, align 8
  %add = fadd double %18, %mul14
  store double %add, double* %arrayidx18, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body8
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %19 = load i32, i32* %k, align 4
  %inc = add nsw i32 %19, 1
  store i32 %inc, i32* %k, align 4
  br label %for.cond6, !llvm.loop !2

for.end:                                          ; preds = %for.cond6
  store double 0.000000e+00, double* %temp2, align 8
  store i32 0, i32* %k, align 4
  br label %for.cond19

for.cond19:                                       ; preds = %for.inc35, %for.end
  %20 = load i32, i32* %k, align 4
  %cmp20 = icmp slt i32 %20, 60
  br i1 %cmp20, label %for.body21, label %for.end37

for.body21:                                       ; preds = %for.cond19
  %21 = load i32, i32* %k, align 4
  %22 = load i32, i32* %i, align 4
  %cmp22 = icmp slt i32 %21, %22
  br i1 %cmp22, label %if.then23, label %if.end34

if.then23:                                        ; preds = %for.body21
  %23 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %24 = load i32, i32* %k, align 4
  %idxprom24 = sext i32 %24 to i64
  %arrayidx25 = getelementptr inbounds [80 x double], [80 x double]* %23, i64 %idxprom24
  %25 = load i32, i32* %j, align 4
  %idxprom26 = sext i32 %25 to i64
  %arrayidx27 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx25, i64 0, i64 %idxprom26
  %26 = load double, double* %arrayidx27, align 8
  %27 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %28 = load i32, i32* %i, align 4
  %idxprom28 = sext i32 %28 to i64
  %arrayidx29 = getelementptr inbounds [60 x double], [60 x double]* %27, i64 %idxprom28
  %29 = load i32, i32* %k, align 4
  %idxprom30 = sext i32 %29 to i64
  %arrayidx31 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx29, i64 0, i64 %idxprom30
  %30 = load double, double* %arrayidx31, align 8
  %mul32 = fmul double %26, %30
  %31 = load double, double* %temp2, align 8
  %add33 = fadd double %31, %mul32
  store double %add33, double* %temp2, align 8
  br label %if.end34

if.end34:                                         ; preds = %if.then23, %for.body21
  br label %for.inc35

for.inc35:                                        ; preds = %if.end34
  %32 = load i32, i32* %k, align 4
  %inc36 = add nsw i32 %32, 1
  store i32 %inc36, i32* %k, align 4
  br label %for.cond19, !llvm.loop !4

for.end37:                                        ; preds = %for.cond19
  %33 = load double, double* %beta.addr, align 8
  %34 = load [80 x double]*, [80 x double]** %C.addr, align 8
  %35 = load i32, i32* %i, align 4
  %idxprom38 = sext i32 %35 to i64
  %arrayidx39 = getelementptr inbounds [80 x double], [80 x double]* %34, i64 %idxprom38
  %36 = load i32, i32* %j, align 4
  %idxprom40 = sext i32 %36 to i64
  %arrayidx41 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx39, i64 0, i64 %idxprom40
  %37 = load double, double* %arrayidx41, align 8
  %mul42 = fmul double %33, %37
  %38 = load double, double* %alpha.addr, align 8
  %39 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %40 = load i32, i32* %i, align 4
  %idxprom43 = sext i32 %40 to i64
  %arrayidx44 = getelementptr inbounds [80 x double], [80 x double]* %39, i64 %idxprom43
  %41 = load i32, i32* %j, align 4
  %idxprom45 = sext i32 %41 to i64
  %arrayidx46 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx44, i64 0, i64 %idxprom45
  %42 = load double, double* %arrayidx46, align 8
  %mul47 = fmul double %38, %42
  %43 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %44 = load i32, i32* %i, align 4
  %idxprom48 = sext i32 %44 to i64
  %arrayidx49 = getelementptr inbounds [60 x double], [60 x double]* %43, i64 %idxprom48
  %45 = load i32, i32* %i, align 4
  %idxprom50 = sext i32 %45 to i64
  %arrayidx51 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx49, i64 0, i64 %idxprom50
  %46 = load double, double* %arrayidx51, align 8
  %mul52 = fmul double %mul47, %46
  %add53 = fadd double %mul42, %mul52
  %47 = load double, double* %alpha.addr, align 8
  %48 = load double, double* %temp2, align 8
  %mul54 = fmul double %47, %48
  %add55 = fadd double %add53, %mul54
  %49 = load [80 x double]*, [80 x double]** %C.addr, align 8
  %50 = load i32, i32* %i, align 4
  %idxprom56 = sext i32 %50 to i64
  %arrayidx57 = getelementptr inbounds [80 x double], [80 x double]* %49, i64 %idxprom56
  %51 = load i32, i32* %j, align 4
  %idxprom58 = sext i32 %51 to i64
  %arrayidx59 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx57, i64 0, i64 %idxprom58
  store double %add55, double* %arrayidx59, align 8
  br label %for.inc60

for.inc60:                                        ; preds = %for.end37
  %52 = load i32, i32* %j, align 4
  %inc61 = add nsw i32 %52, 1
  store i32 %inc61, i32* %j, align 4
  br label %for.cond1, !llvm.loop !5

for.end62:                                        ; preds = %for.cond1
  br label %for.inc63

for.inc63:                                        ; preds = %for.end62
  %53 = load i32, i32* %i, align 4
  %inc64 = add nsw i32 %53, 1
  store i32 %inc64, i32* %i, align 4
  br label %for.cond, !llvm.loop !6

for.end65:                                        ; preds = %for.cond
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
