; ModuleID = 'gemver-medium.c'
source_filename = "gemver-medium.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @kernel_gemver(i32 %n, double %alpha, double %beta, [400 x double]* %A, double* %u1, double* %v1, double* %u2, double* %v2, double* %w, double* %x, double* %y, double* %z) #0 {
entry:
  %n.addr = alloca i32, align 4
  %alpha.addr = alloca double, align 8
  %beta.addr = alloca double, align 8
  %A.addr = alloca [400 x double]*, align 8
  %u1.addr = alloca double*, align 8
  %v1.addr = alloca double*, align 8
  %u2.addr = alloca double*, align 8
  %v2.addr = alloca double*, align 8
  %w.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %y.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  store double %alpha, double* %alpha.addr, align 8
  store double %beta, double* %beta.addr, align 8
  store [400 x double]* %A, [400 x double]** %A.addr, align 8
  store double* %u1, double** %u1.addr, align 8
  store double* %v1, double** %v1.addr, align 8
  store double* %u2, double** %u2.addr, align 8
  store double* %v2, double** %v2.addr, align 8
  store double* %w, double** %w.addr, align 8
  store double* %x, double** %x.addr, align 8
  store double* %y, double** %y.addr, align 8
  store double* %z, double** %z.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc16, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 400
  br i1 %cmp, label %for.body, label %for.end18

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 400
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load double*, double** %u1.addr, align 8
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds double, double* %2, i64 %idxprom
  %4 = load double, double* %arrayidx, align 8
  %5 = load double*, double** %v1.addr, align 8
  %6 = load i32, i32* %j, align 4
  %idxprom4 = sext i32 %6 to i64
  %arrayidx5 = getelementptr inbounds double, double* %5, i64 %idxprom4
  %7 = load double, double* %arrayidx5, align 8
  %mul = fmul double %4, %7
  %8 = load double*, double** %u2.addr, align 8
  %9 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %9 to i64
  %arrayidx7 = getelementptr inbounds double, double* %8, i64 %idxprom6
  %10 = load double, double* %arrayidx7, align 8
  %11 = load double*, double** %v2.addr, align 8
  %12 = load i32, i32* %j, align 4
  %idxprom8 = sext i32 %12 to i64
  %arrayidx9 = getelementptr inbounds double, double* %11, i64 %idxprom8
  %13 = load double, double* %arrayidx9, align 8
  %mul10 = fmul double %10, %13
  %add = fadd double %mul, %mul10
  %14 = load [400 x double]*, [400 x double]** %A.addr, align 8
  %15 = load i32, i32* %i, align 4
  %idxprom11 = sext i32 %15 to i64
  %arrayidx12 = getelementptr inbounds [400 x double], [400 x double]* %14, i64 %idxprom11
  %16 = load i32, i32* %j, align 4
  %idxprom13 = sext i32 %16 to i64
  %arrayidx14 = getelementptr inbounds [400 x double], [400 x double]* %arrayidx12, i64 0, i64 %idxprom13
  %17 = load double, double* %arrayidx14, align 8
  %add15 = fadd double %17, %add
  store double %add15, double* %arrayidx14, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %18 = load i32, i32* %j, align 4
  %inc = add nsw i32 %18, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond1, !llvm.loop !4

for.end:                                          ; preds = %for.cond1
  br label %for.inc16

for.inc16:                                        ; preds = %for.end
  %19 = load i32, i32* %i, align 4
  %inc17 = add nsw i32 %19, 1
  store i32 %inc17, i32* %i, align 4
  br label %for.cond, !llvm.loop !6

for.end18:                                        ; preds = %for.cond
  store i32 0, i32* %i, align 4
  br label %for.cond19

for.cond19:                                       ; preds = %for.inc39, %for.end18
  %20 = load i32, i32* %i, align 4
  %cmp20 = icmp slt i32 %20, 400
  br i1 %cmp20, label %for.body21, label %for.end41

for.body21:                                       ; preds = %for.cond19
  store i32 0, i32* %j, align 4
  br label %for.cond22

for.cond22:                                       ; preds = %for.inc36, %for.body21
  %21 = load i32, i32* %j, align 4
  %cmp23 = icmp slt i32 %21, 400
  br i1 %cmp23, label %for.body24, label %for.end38

for.body24:                                       ; preds = %for.cond22
  %22 = load double, double* %beta.addr, align 8
  %23 = load [400 x double]*, [400 x double]** %A.addr, align 8
  %24 = load i32, i32* %j, align 4
  %idxprom25 = sext i32 %24 to i64
  %arrayidx26 = getelementptr inbounds [400 x double], [400 x double]* %23, i64 %idxprom25
  %25 = load i32, i32* %i, align 4
  %idxprom27 = sext i32 %25 to i64
  %arrayidx28 = getelementptr inbounds [400 x double], [400 x double]* %arrayidx26, i64 0, i64 %idxprom27
  %26 = load double, double* %arrayidx28, align 8
  %mul29 = fmul double %22, %26
  %27 = load double*, double** %y.addr, align 8
  %28 = load i32, i32* %j, align 4
  %idxprom30 = sext i32 %28 to i64
  %arrayidx31 = getelementptr inbounds double, double* %27, i64 %idxprom30
  %29 = load double, double* %arrayidx31, align 8
  %mul32 = fmul double %mul29, %29
  %30 = load double*, double** %x.addr, align 8
  %31 = load i32, i32* %i, align 4
  %idxprom33 = sext i32 %31 to i64
  %arrayidx34 = getelementptr inbounds double, double* %30, i64 %idxprom33
  %32 = load double, double* %arrayidx34, align 8
  %add35 = fadd double %32, %mul32
  store double %add35, double* %arrayidx34, align 8
  br label %for.inc36

for.inc36:                                        ; preds = %for.body24
  %33 = load i32, i32* %j, align 4
  %inc37 = add nsw i32 %33, 1
  store i32 %inc37, i32* %j, align 4
  br label %for.cond22, !llvm.loop !7

for.end38:                                        ; preds = %for.cond22
  br label %for.inc39

for.inc39:                                        ; preds = %for.end38
  %34 = load i32, i32* %i, align 4
  %inc40 = add nsw i32 %34, 1
  store i32 %inc40, i32* %i, align 4
  br label %for.cond19, !llvm.loop !8

for.end41:                                        ; preds = %for.cond19
  store i32 0, i32* %i, align 4
  br label %for.cond42

for.cond42:                                       ; preds = %for.inc52, %for.end41
  %35 = load i32, i32* %i, align 4
  %cmp43 = icmp slt i32 %35, 400
  br i1 %cmp43, label %for.body44, label %for.end54

for.body44:                                       ; preds = %for.cond42
  %36 = load double*, double** %x.addr, align 8
  %37 = load i32, i32* %i, align 4
  %idxprom45 = sext i32 %37 to i64
  %arrayidx46 = getelementptr inbounds double, double* %36, i64 %idxprom45
  %38 = load double, double* %arrayidx46, align 8
  %39 = load double*, double** %z.addr, align 8
  %40 = load i32, i32* %i, align 4
  %idxprom47 = sext i32 %40 to i64
  %arrayidx48 = getelementptr inbounds double, double* %39, i64 %idxprom47
  %41 = load double, double* %arrayidx48, align 8
  %add49 = fadd double %38, %41
  %42 = load double*, double** %x.addr, align 8
  %43 = load i32, i32* %i, align 4
  %idxprom50 = sext i32 %43 to i64
  %arrayidx51 = getelementptr inbounds double, double* %42, i64 %idxprom50
  store double %add49, double* %arrayidx51, align 8
  br label %for.inc52

for.inc52:                                        ; preds = %for.body44
  %44 = load i32, i32* %i, align 4
  %inc53 = add nsw i32 %44, 1
  store i32 %inc53, i32* %i, align 4
  br label %for.cond42, !llvm.loop !9

for.end54:                                        ; preds = %for.cond42
  store i32 0, i32* %i, align 4
  br label %for.cond55

for.cond55:                                       ; preds = %for.inc75, %for.end54
  %45 = load i32, i32* %i, align 4
  %cmp56 = icmp slt i32 %45, 400
  br i1 %cmp56, label %for.body57, label %for.end77

for.body57:                                       ; preds = %for.cond55
  store i32 0, i32* %j, align 4
  br label %for.cond58

for.cond58:                                       ; preds = %for.inc72, %for.body57
  %46 = load i32, i32* %j, align 4
  %cmp59 = icmp slt i32 %46, 400
  br i1 %cmp59, label %for.body60, label %for.end74

for.body60:                                       ; preds = %for.cond58
  %47 = load double, double* %alpha.addr, align 8
  %48 = load [400 x double]*, [400 x double]** %A.addr, align 8
  %49 = load i32, i32* %i, align 4
  %idxprom61 = sext i32 %49 to i64
  %arrayidx62 = getelementptr inbounds [400 x double], [400 x double]* %48, i64 %idxprom61
  %50 = load i32, i32* %j, align 4
  %idxprom63 = sext i32 %50 to i64
  %arrayidx64 = getelementptr inbounds [400 x double], [400 x double]* %arrayidx62, i64 0, i64 %idxprom63
  %51 = load double, double* %arrayidx64, align 8
  %mul65 = fmul double %47, %51
  %52 = load double*, double** %x.addr, align 8
  %53 = load i32, i32* %j, align 4
  %idxprom66 = sext i32 %53 to i64
  %arrayidx67 = getelementptr inbounds double, double* %52, i64 %idxprom66
  %54 = load double, double* %arrayidx67, align 8
  %mul68 = fmul double %mul65, %54
  %55 = load double*, double** %w.addr, align 8
  %56 = load i32, i32* %i, align 4
  %idxprom69 = sext i32 %56 to i64
  %arrayidx70 = getelementptr inbounds double, double* %55, i64 %idxprom69
  %57 = load double, double* %arrayidx70, align 8
  %add71 = fadd double %57, %mul68
  store double %add71, double* %arrayidx70, align 8
  br label %for.inc72

for.inc72:                                        ; preds = %for.body60
  %58 = load i32, i32* %j, align 4
  %inc73 = add nsw i32 %58, 1
  store i32 %inc73, i32* %j, align 4
  br label %for.cond58, !llvm.loop !10

for.end74:                                        ; preds = %for.cond58
  br label %for.inc75

for.inc75:                                        ; preds = %for.end74
  %59 = load i32, i32* %i, align 4
  %inc76 = add nsw i32 %59, 1
  store i32 %inc76, i32* %i, align 4
  br label %for.cond55, !llvm.loop !11

for.end77:                                        ; preds = %for.cond55
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"Ubuntu clang version 13.0.1-++20220120110844+75e33f71c2da-1~exp1~20220120230854.66"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
!6 = distinct !{!6, !5}
!7 = distinct !{!7, !5}
!8 = distinct !{!8, !5}
!9 = distinct !{!9, !5}
!10 = distinct !{!10, !5}
!11 = distinct !{!11, !5}
