; ModuleID = 'symm.c'
source_filename = "symm.c"
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
  %temp2 = alloca double, align 8
  store double %alpha, double* %alpha.addr, align 8
  store double %beta, double* %beta.addr, align 8
  store [80 x double]* %C, [80 x double]** %C.addr, align 8
  store [60 x double]* %A, [60 x double]** %A.addr, align 8
  store [80 x double]* %B, [80 x double]** %B.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc54, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 60
  br i1 %cmp, label %for.body, label %for.end56

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc51, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 80
  br i1 %cmp2, label %for.body3, label %for.end53

for.body3:                                        ; preds = %for.cond1
  store double 0.000000e+00, double* %temp2, align 8
  store i32 0, i32* %k, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %2 = load i32, i32* %k, align 4
  %cmp5 = icmp slt i32 %2, 60
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %3 = load i32, i32* %k, align 4
  %4 = load i32, i32* %i, align 4
  %cmp7 = icmp slt i32 %3, %4
  br i1 %cmp7, label %if.then, label %if.end

if.then:                                          ; preds = %for.body6
  %5 = load double, double* %alpha.addr, align 8
  %6 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %7 = load i32, i32* %i, align 4
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds [80 x double], [80 x double]* %6, i64 %idxprom
  %8 = load i32, i32* %j, align 4
  %idxprom8 = sext i32 %8 to i64
  %arrayidx9 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx, i64 0, i64 %idxprom8
  %9 = load double, double* %arrayidx9, align 8
  %mul = fmul double %5, %9
  %10 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %11 = load i32, i32* %i, align 4
  %idxprom10 = sext i32 %11 to i64
  %arrayidx11 = getelementptr inbounds [60 x double], [60 x double]* %10, i64 %idxprom10
  %12 = load i32, i32* %k, align 4
  %idxprom12 = sext i32 %12 to i64
  %arrayidx13 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx11, i64 0, i64 %idxprom12
  %13 = load double, double* %arrayidx13, align 8
  %mul14 = fmul double %mul, %13
  %14 = load [80 x double]*, [80 x double]** %C.addr, align 8
  %15 = load i32, i32* %k, align 4
  %idxprom15 = sext i32 %15 to i64
  %arrayidx16 = getelementptr inbounds [80 x double], [80 x double]* %14, i64 %idxprom15
  %16 = load i32, i32* %j, align 4
  %idxprom17 = sext i32 %16 to i64
  %arrayidx18 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx16, i64 0, i64 %idxprom17
  %17 = load double, double* %arrayidx18, align 8
  %add = fadd double %17, %mul14
  store double %add, double* %arrayidx18, align 8
  %18 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %19 = load i32, i32* %k, align 4
  %idxprom19 = sext i32 %19 to i64
  %arrayidx20 = getelementptr inbounds [80 x double], [80 x double]* %18, i64 %idxprom19
  %20 = load i32, i32* %j, align 4
  %idxprom21 = sext i32 %20 to i64
  %arrayidx22 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx20, i64 0, i64 %idxprom21
  %21 = load double, double* %arrayidx22, align 8
  %22 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %23 = load i32, i32* %i, align 4
  %idxprom23 = sext i32 %23 to i64
  %arrayidx24 = getelementptr inbounds [60 x double], [60 x double]* %22, i64 %idxprom23
  %24 = load i32, i32* %k, align 4
  %idxprom25 = sext i32 %24 to i64
  %arrayidx26 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx24, i64 0, i64 %idxprom25
  %25 = load double, double* %arrayidx26, align 8
  %mul27 = fmul double %21, %25
  %26 = load double, double* %temp2, align 8
  %add28 = fadd double %26, %mul27
  store double %add28, double* %temp2, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body6
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %27 = load i32, i32* %k, align 4
  %inc = add nsw i32 %27, 1
  store i32 %inc, i32* %k, align 4
  br label %for.cond4, !llvm.loop !2

for.end:                                          ; preds = %for.cond4
  %28 = load double, double* %beta.addr, align 8
  %29 = load [80 x double]*, [80 x double]** %C.addr, align 8
  %30 = load i32, i32* %i, align 4
  %idxprom29 = sext i32 %30 to i64
  %arrayidx30 = getelementptr inbounds [80 x double], [80 x double]* %29, i64 %idxprom29
  %31 = load i32, i32* %j, align 4
  %idxprom31 = sext i32 %31 to i64
  %arrayidx32 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx30, i64 0, i64 %idxprom31
  %32 = load double, double* %arrayidx32, align 8
  %mul33 = fmul double %28, %32
  %33 = load double, double* %alpha.addr, align 8
  %34 = load [80 x double]*, [80 x double]** %B.addr, align 8
  %35 = load i32, i32* %i, align 4
  %idxprom34 = sext i32 %35 to i64
  %arrayidx35 = getelementptr inbounds [80 x double], [80 x double]* %34, i64 %idxprom34
  %36 = load i32, i32* %j, align 4
  %idxprom36 = sext i32 %36 to i64
  %arrayidx37 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx35, i64 0, i64 %idxprom36
  %37 = load double, double* %arrayidx37, align 8
  %mul38 = fmul double %33, %37
  %38 = load [60 x double]*, [60 x double]** %A.addr, align 8
  %39 = load i32, i32* %i, align 4
  %idxprom39 = sext i32 %39 to i64
  %arrayidx40 = getelementptr inbounds [60 x double], [60 x double]* %38, i64 %idxprom39
  %40 = load i32, i32* %i, align 4
  %idxprom41 = sext i32 %40 to i64
  %arrayidx42 = getelementptr inbounds [60 x double], [60 x double]* %arrayidx40, i64 0, i64 %idxprom41
  %41 = load double, double* %arrayidx42, align 8
  %mul43 = fmul double %mul38, %41
  %add44 = fadd double %mul33, %mul43
  %42 = load double, double* %alpha.addr, align 8
  %43 = load double, double* %temp2, align 8
  %mul45 = fmul double %42, %43
  %add46 = fadd double %add44, %mul45
  %44 = load [80 x double]*, [80 x double]** %C.addr, align 8
  %45 = load i32, i32* %i, align 4
  %idxprom47 = sext i32 %45 to i64
  %arrayidx48 = getelementptr inbounds [80 x double], [80 x double]* %44, i64 %idxprom47
  %46 = load i32, i32* %j, align 4
  %idxprom49 = sext i32 %46 to i64
  %arrayidx50 = getelementptr inbounds [80 x double], [80 x double]* %arrayidx48, i64 0, i64 %idxprom49
  store double %add46, double* %arrayidx50, align 8
  br label %for.inc51

for.inc51:                                        ; preds = %for.end
  %47 = load i32, i32* %j, align 4
  %inc52 = add nsw i32 %47, 1
  store i32 %inc52, i32* %j, align 4
  br label %for.cond1, !llvm.loop !4

for.end53:                                        ; preds = %for.cond1
  br label %for.inc54

for.inc54:                                        ; preds = %for.end53
  %48 = load i32, i32* %i, align 4
  %inc55 = add nsw i32 %48, 1
  store i32 %inc55, i32* %i, align 4
  br label %for.cond, !llvm.loop !5

for.end56:                                        ; preds = %for.cond
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
