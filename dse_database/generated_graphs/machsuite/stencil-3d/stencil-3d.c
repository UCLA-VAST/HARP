#pragma ACCEL kernel

void stencil3d(long C0,long C1,long orig[39304],long sol[32768])
{
  long sum0;
  long sum1;
  long mul0;
  long mul1;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (long i = 1; i < 33; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (long j = 1; j < 33; j++) {
/* Standardize from: for(long ko =(long )1;ko <((long )(34 - 1));ko +=((long )1)) {...} */
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (long ko = 0; ko <= 31; ko++) {
        long _in_ko = 1L + 1L * ko;
// #define UNROLL_IMPL(k)                                         \
//           sum0 = orig[INDX(row_size, col_size, k, j, i)];      \
//           sum1 = orig[INDX(row_size, col_size, k, j, i + 1)] + \
//                  orig[INDX(row_size, col_size, k, j, i - 1)] + \
//                  orig[INDX(row_size, col_size, k, j + 1, i)] + \
//                  orig[INDX(row_size, col_size, k, j - 1, i)] + \
//                  orig[INDX(row_size, col_size, k + 1, j, i)] + \
//                  orig[INDX(row_size, col_size, k - 1, j, i)];  \
//           mul0 = sum0 * C0;                                    \
//           mul1 = sum1 * C1;                                    \
//           sol[INDX(row_size, col_size, k, j, i)] = mul0 + mul1;
        sum0 = orig[_in_ko + ((long )0) + ((long )34) * (j + ((long )34) * i)];
        sum1 = orig[_in_ko + ((long )0) + ((long )34) * (j + ((long )34) * (i + ((long )1)))] + orig[_in_ko + ((long )0) + ((long )34) * (j + ((long )34) * (i - ((long )1)))] + orig[_in_ko + ((long )0) + ((long )34) * (j + ((long )1) + ((long )34) * i)] + orig[_in_ko + ((long )0) + ((long )34) * (j - ((long )1) + ((long )34) * i)] + orig[_in_ko + ((long )0) + ((long )1) + ((long )34) * (j + ((long )34) * i)] + orig[_in_ko + ((long )0) - ((long )1) + ((long )34) * (j + ((long )34) * i)];
        mul0 = sum0 * C0;
        mul1 = sum1 * C1;
        sol[_in_ko + ((long )0) + ((long )34) * (j + ((long )34) * i)] = mul0 + mul1;
      }
    }
  }
}
