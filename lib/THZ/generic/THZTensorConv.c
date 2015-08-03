/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorConv.c"
#else

/*
  2D Input, 2D kernel  : convolve given image with the given kernel.
*/
THZ_API void THZTensor_(validXCorr2Dptr)(real *r_,
                                       real alpha,
                                       real *t_, long ir, long ic,
                                       real *k_, long kr, long kc,
                                       long sr, long sc)
{
  long or = (ir - kr) / sr + 1;
  long oc = (ic - kc) / sc + 1;

  long xx, yy, kx, ky;

  if ((sc != 1) || (oc < 4))  {
    /* regular convolution */
    for(yy = 0; yy < or; yy++) {
      for(xx = 0; xx < oc; xx++) {
        /* Dot product in two dimensions... (between input image and the mask) */
        real *pi_ = t_ + yy*sr*ic + xx*sc;
        real *pw_ = k_;
        real sum = 0;
        for(ky = 0; ky < kr; ky++) {
          for(kx = 0; kx < kc; kx++) {
            sum += pi_[kx]*pw_[kx];
          }
          pi_ += ic; /* next input line */
          pw_ += kc; /* next mask line */
        }
        /* Update output */
        *r_++ += alpha*sum;
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < or; yy++) {
      real *pi_ = t_ + yy*sr*ic;
      real *pw_ = k_;
      for (ky = 0; ky < kr; ky++) {
        real *pis_ = pi_;
        for (kx = 0; kx < kc; kx++) {
          THZVector_(add)(r_, pis_, alpha*pw_[kx], oc);
          pis_++;
        }
        pi_ += ic; /* next input line */
        pw_ += kc; /* next mask line */
      }
      r_ += oc;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel.
*/
THZ_API void THZTensor_(validConv2Dptr)(real *r_,
                                      real alpha,
                                      real *t_, long ir, long ic,
                                      real *k_, long kr, long kc,
                                      long sr, long sc)
{
  long or = (ir - kr) / sr + 1;
  long oc = (ic - kc) / sc + 1;

  long xx, yy, kx, ky;

  if ((sc != 1) || (oc < 4))  {
    /* regular convolution */
    for(yy = 0; yy < or; yy++) {
      for(xx = 0; xx < oc; xx++) {
        /* Dot product in two dimensions... (between input image and the mask) */
        real *pi_ = t_ + yy*sr*ic + xx*sc;
        real *pw_ = k_ + kr*kc - 1;
        real sum = 0;
        for(ky = 0; ky < kr; ky++) {
          for(kx = 0; kx < kc; kx++) {
            sum += pi_[kx]*pw_[-kx];
          }
          pi_ += ic; /* next input line */
          pw_ -= kc; /* next mask line */
        }
        /* Update output */
        *r_++ += alpha*sum;
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < or; yy++) {
      real *pw_ = k_ + kr*kc - 1;
      real *pi_ = t_ + yy*sr*ic;
      for (ky = 0; ky < kr; ky++) {
        real *pis_ = pi_;
        for (kx = 0; kx < kc; kx++) {
          THZVector_(add)(r_, pis_, alpha*pw_[-kx], oc);
          pis_++;
        }
        pi_ += ic; /* next input line */
        pw_ -= kc; /* next mask line */
      }
      r_ += oc;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel, full convolution.
*/
THZ_API void THZTensor_(fullConv2Dptr)(real *r_,
                                     real alpha,
                                     real *t_, long ir, long ic,
                                     real *k_, long kr, long kc,
                                     long sr, long sc)
{
  long oc = (ic - 1) * sc + kc;

  long xx, yy, kx, ky;

  if ((sc != 1) || (ic < 4))  {
    /* regular convolution */
    for(yy = 0; yy < ir; yy++) {
      for(xx = 0; xx < ic; xx++) {
        /* Outer product in two dimensions... (between input image and the mask) */
        real *po_ = r_ + yy*sr*oc + xx*sc;
        real *pw_ = k_;
        for(ky = 0; ky < kr; ky++)
        {
          real z = *t_ * alpha;
          for(kx = 0; kx < kc; kx++) {
            po_[kx] += z * pw_[kx];
          }
          po_ += oc; /* next input line */
          pw_ += kc; /* next mask line */
        }
        t_++;
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < ir; yy++) {
      real *po_ = r_ + yy*sr*oc;
      real *pw_ = k_;
      for (ky = 0; ky < kr; ky++) {
        real *pos_ = po_;
        for (kx = 0; kx < kc; kx++) {
          THZVector_(add)(pos_, t_, alpha*pw_[kx], ic);
          pos_++;
        }
        po_ += oc; /* next input line */
        pw_ += kc; /* next mask line */
      }
      t_ += ic;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel, full convolution.
*/
THZ_API void THZTensor_(fullXCorr2Dptr)(real *r_,
                                      real alpha,
                                      real *t_, long ir, long ic,
                                      real *k_, long kr, long kc,
                                      long sr, long sc)
{
  long oc = (ic - 1) * sc + kc;

  long xx, yy, kx, ky;

  if ((sc != 1) || (ic < 4))  {
    /* regular convolution */
    for(yy = 0; yy < ir; yy++) {
      for(xx = 0; xx < ic; xx++) {
        /* Outer product in two dimensions... (between input image and the mask) */
        real *po_ = r_ + yy*sr*oc + xx*sc;
        real *pw_ = k_ + kr*kc -1;
        long kx, ky;
        for(ky = 0; ky < kr; ky++)
        {
          real z = *t_ * alpha;
          for(kx = 0; kx < kc; kx++) {
            po_[kx] += z * pw_[-kx];
          }
          po_ += oc; /* next input line */
          pw_ -= kc; /* next mask line */
        }
        t_++;
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < ir; yy++) {
      real *po_ = r_ + yy*sr*oc;
      real *pw_ = k_ + kr*kc -1;
      for (ky = 0; ky < kr; ky++) {
        real *pos_ = po_;
        for (kx = 0; kx < kc; kx++) {
          THZVector_(add)(pos_, t_, pw_[-kx]*alpha, ic);
          pos_++;
        }
        po_ += oc; /* next input line */
        pw_ -= kc; /* next mask line */
      }
      t_ += ic;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel, valid convolution.
  for sr,sc=1 this is equivalent to validXCorr2Dptr, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
THZ_API void THZTensor_(validXCorr2DRevptr)(real *r_,
                                          real alpha,
                                          real *t_, long ir, long ic,
                                          real *k_, long kr, long kc,
                                          long sr, long sc)
{
  long or = ir - (kr - 1) * sr;
  long oc = ic - (kc - 1) * sc;

  long xx, yy, kx, ky;

  if ((sc != 1) || (kc < 4))  {
    /* regular convolution */
    for(yy = 0; yy < kr; yy++) {
      for(xx = 0; xx < kc; xx++) {
        real *po_ = r_;
        real *pi_ = t_ + yy*sr*ic + xx*sc;
        real z = *k_++ * alpha;

        for(ky = 0; ky < or; ky++) {
          for(kx = 0; kx < oc; kx++)
            po_[kx] += z * pi_[kx];
          pi_ += ic;
          po_ += oc;
        }
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < kr; yy++) {
      for(xx = 0; xx < kc; xx++) {
        real *po_ = r_;
        real *pi_ = t_ + yy*sr*ic + xx*sc;
        real z = *k_++ * alpha;

        for(ky = 0; ky < or; ky++) {
          THZVector_(add)(po_, pi_, z, oc);
          pi_ += ic;
          po_ += oc;
        }
      }
    }
  }
}
/*
  3D Input, 3D kernel  : convolve given volume with the given kernel.
*/
THZ_API void THZTensor_(validXCorr3Dptr)(real *r_,
                                       real alpha,
                                       real *t_, long it, long ir, long ic,
                                       real *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc)
{
  long ot = (it - kt) / st + 1;
  long or = (ir - kr) / sr + 1;
  long oc = (ic - kc) / sc + 1;

  long zz, xx, yy;

  for (zz = 0; zz < ot; zz++)
  {
    for(yy = 0; yy < or; yy++)
    {
      for(xx = 0; xx < oc; xx++)
      {
        /* Dot product in two dimensions... (between input image and the mask) */
        real *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
        real *pw_ = k_;
        real sum = 0;
        long kz, kx, ky;
        for(kz = 0; kz < kt; kz++)
        {
          for(ky = 0; ky < kr; ky++)
          {
            for(kx = 0; kx < kc; kx++) {
              sum += pi_[kx]*pw_[kx];
            }
            pi_ += ic; /* next input line */
            pw_ += kc; /* next mask line */
          }
          pi_ += (ir-kr)*ic; /* next input slice */
        }
        /* Update output */
        *r_++ += sum*alpha;
      }
    }
  }
}

/*
  3D Input, 3D kernel  : convolve given volume with the given kernel.
*/
THZ_API void THZTensor_(validConv3Dptr)(real *r_,
                                      real alpha,
                                      real *t_, long it, long ir, long ic,
                                      real *k_, long kt, long kr, long kc,
                                      long st, long sr, long sc)
{
  long ot = (it - kt) / st + 1;
  long or = (ir - kr) / sr + 1;
  long oc = (ic - kc) / sc + 1;

  long zz, xx, yy;

  for(zz = 0; zz < ot; zz++)
  {
    for(yy = 0; yy < or; yy++)
    {
      for(xx = 0; xx < oc; xx++)
      {
        /* Dot product in two dimensions... (between input image and the mask) */
        real *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
        real *pw_ = k_ + kt*kr*kc - 1;
        real sum = 0;
        long kz, kx, ky;
        for(kz = 0; kz < kt; kz++)
        {
          for(ky = 0; ky < kr; ky++)
          {
            for(kx = 0; kx < kc; kx++) {
              sum += pi_[kx]*pw_[-kx];
            }
            pi_ += ic; /* next input line */
            pw_ -= kc; /* next mask line */
          }
          pi_ += (ir-kr)*ic; /* next input slice */
        }
        /* Update output */
        *r_++ += alpha*sum;
      }
    }
  }
}


/*
  3D Input, 3D kernel  : convolve given volume with the given kernel, full convolution.
*/
THZ_API void THZTensor_(fullConv3Dptr)(real *r_,
                                     real alpha,
                                     real *t_, long it, long ir, long ic,
                                     real *k_, long kt, long kr, long kc,
                                     long st, long sr, long sc)
{
  long or = (ir - 1) * sr + kr;
  long oc = (ic - 1) * sc + kc;

  long zz, xx, yy;

  for(zz = 0; zz < it; zz++)
  {
    for(yy = 0; yy < ir; yy++)
    {
      for(xx = 0; xx < ic; xx++)
      {
        /* Outer product in two dimensions... (between input image and the mask) */
        real *po_ = r_ + zz*st*or*oc + yy*sr*oc + xx*sc;
        real *pw_ = k_;
        long kz, kx, ky;
        /* printf("Output Plane : %ld,%ld,%ld, input val=%g\n",zz,yy,xx,*t_); */
        for(kz = 0; kz < kt; kz++)
        {
          for(ky = 0; ky < kr; ky++)
          {
            real z = *t_ * alpha;
            for(kx = 0; kx < kc; kx++) {
              /* printf("o=%g,k=%g," , po_[kx],pw_[kx]); */
              po_[kx] += z * pw_[kx];
              /* printf("o=%g " , po_[kx]); */
            }
            /* printf("\n"); */
            po_ += oc; /* next input line */
            pw_ += kc; /* next mask line */
          }
          po_ += (or-kr)*oc; /* next output slice */
          /* printf("\n"); */
        }
        t_++;
      }
    }
  }
}

/*
  3D Input, 3D kernel  : convolve given volume with the given kernel, full convolution.
*/
THZ_API void THZTensor_(fullXCorr3Dptr)(real *r_,
                                      real alpha,
                                      real *t_, long it, long ir, long ic,
                                      real *k_, long kt, long kr, long kc,
                                      long st, long sr, long sc)
{
  long or = (ir - 1) * sr + kr;
  long oc = (ic - 1) * sc + kc;

  long zz, xx, yy;

  for(zz = 0; zz < it; zz++)
  {
    for(yy = 0; yy < ir; yy++)
    {
      for(xx = 0; xx < ic; xx++)
      {
        /* Outer product in two dimensions... (between input image and the mask) */
        real *po_ = r_ + zz*st*or*oc + yy*sr*oc + xx*sc;
        real *pw_ = k_ + kt*kr*kc -1;
        long kz, kx, ky;
        for(kz = 0; kz < kt; kz++)
        {
          for(ky = 0; ky < kr; ky++)
          {
            real z = *t_ * alpha;
            for(kx = 0; kx < kc; kx++) {
              po_[kx] += z * pw_[-kx];
            }
            po_ += oc; /* next input line */
            pw_ -= kc; /* next mask line */
          }
          po_ += (or-kr)*oc; /* next output slice */
        }
        t_++;
      }
    }
  }
}

/*
  3D Input, 3D kernel  : convolve given image with the given kernel, valid convolution.
  for sr,sc=1 this is equivalent to validXCorr3Dptr, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
THZ_API void THZTensor_(validXCorr3DRevptr)(real *r_,
                                          real alpha,
                                          real *t_, long it, long ir, long ic,
                                          real *k_, long kt, long kr, long kc,
                                          long st, long sr, long sc)
{
  long ot = it - (kt - 1) * st;
  long or = ir - (kr - 1) * sr;
  long oc = ic - (kc - 1) * sc;

  long zz, xx, yy;
  for(zz = 0; zz < kt; zz++)
  {
    for(yy = 0; yy < kr; yy++)
    {
      for(xx = 0; xx < kc; xx++)
      {
        real *po_ = r_;
        real *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
        real z = *k_++ * alpha;
        long kz, kx, ky;
        for(kz = 0; kz < ot; kz++)
        {
          for(ky = 0; ky < or; ky++)
          {
            for(kx = 0; kx < oc; kx++)
              po_[kx] += z * pi_[kx];
            pi_ += ic;
            po_ += oc;
          }
          pi_ += (ir-or)*ic; /* next input slice */
        }
      }
    }
  }
}

void THZTensor_(conv2d)(real* output_data,
                       real alpha,
                       real* ptr_input, long nInputRows, long nInputCols,
                       real* ptr_weight, long nKernelRows, long nKernelCols,
                       long srow, long scol,
                       const char *vf, const char *xc)
{
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can be 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can be 'X' or 'C'");
  if (*vf == 'F')
    if (*xc == 'X')
      THZTensor_(fullXCorr2Dptr)(output_data,
                                alpha,
                                ptr_input,  nInputRows,  nInputCols,
                                ptr_weight, nKernelRows, nKernelCols,
                                srow, scol);
    else
      THZTensor_(fullConv2Dptr)(output_data,
                               alpha,
                               ptr_input,  nInputRows,  nInputCols,
                               ptr_weight, nKernelRows, nKernelCols,
                               srow, scol);
  else
    if (*xc == 'X')
      THZTensor_(validXCorr2Dptr)(output_data,
                                 alpha,
                                 ptr_input,  nInputRows,  nInputCols,
                                 ptr_weight, nKernelRows, nKernelCols,
                                 srow, scol);
    else
      THZTensor_(validConv2Dptr)(output_data,
                                alpha,
                                ptr_input,  nInputRows,  nInputCols,
                                ptr_weight, nKernelRows, nKernelCols,
                                srow, scol);
}

void THZTensor_(conv3d)(real* output_data,
                       real alpha,
                       real* ptr_input, long nInputDepth, long nInputRows, long nInputCols,
                       real* ptr_weight, long nKernelDepth, long nKernelRows, long nKernelCols,
                       long sdepth, long srow, long scol,
                       const char *vf, const char *xc)
{
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can be 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can be 'X' or 'C'");
  if (*vf == 'F')
    if (*xc == 'X')
      THZTensor_(fullXCorr3Dptr)(output_data,
                                alpha,
                                ptr_input, nInputDepth, nInputRows,  nInputCols,
                                ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                sdepth, srow, scol);
    else
      THZTensor_(fullConv3Dptr)(output_data,
                               alpha,
                               ptr_input, nInputDepth, nInputRows,  nInputCols,
                               ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                               sdepth, srow, scol);
  else
    if (*xc == 'X')
      THZTensor_(validXCorr3Dptr)(output_data,
                                 alpha,
                                 ptr_input, nInputDepth, nInputRows,  nInputCols,
                                 ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                 sdepth, srow, scol);
    else
      THZTensor_(validConv3Dptr)(output_data,
                                alpha,
                                ptr_input, nInputDepth, nInputRows,  nInputCols,
                                ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                sdepth, srow, scol);
}

long THZTensor_(convsize)(long x, long k, long s, const char* vf)
{
  THArgCheck(*vf == 'V' || *vf == 'F', 1, "type of convolution can be 'V' or 'F'");
  if (*vf == 'V')
    return (x-k)/s + 1;
  else
    return (x-1)*s + k;
}


/*
  3D input, 3D kernel, 4D output
  like rank1 update
  A <- xx' + beta*A
  for sr,sc=1 this is equivalent to conv2Dger, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
void THZTensor_(conv2DRevger)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelPlane, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0;
  THZTensor *input;
  THZTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0 = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];
  nOutputPlane = nInputPlane * kernel->size[0];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "covn2DRevger : Input image is smaller than kernel");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize4d)(r_,nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    /*THZTensor_(zero)(r_);*/

#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }
  }
  else if (beta != 1)
  {
    /*THZTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] *= beta;
    }
  }

#pragma omp parallel for private(k)
  for(k = 0; k < nKernelPlane; k++)
  {
    long i;
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get output */
      real *ptr_output = output_data + k*nInputPlane*nOutputCols*nOutputRows + i*nOutputCols*nOutputRows;
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      THZTensor_(validXCorr2DRevptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows; */
    }
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}


/*
  3D input, 3D kernel, 4D output
  like rank1 update
  A <- xx' + beta*A
  for sr,sc=1 this is equivalent to conv2Dger, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
void THZTensor_(conv2DRevgerm)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol)
{
  long nbatch, nInputPlane, nInputRows, nInputCols;
  long nKernelPlane, nKernelRows, nKernelCols;
  long nOutputRows, nOutputCols;
  long istride0, kstride0, istride1, kstride1;
  THZTensor *input;
  THZTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  istride1    = input->stride[1];
  nbatch      = input->size[0];
  nInputPlane = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0 = kernel->stride[0];
  kstride1 = kernel->stride[1];
  nKernelPlane = kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "conv2DRevger : Input image is smaller than kernel");
  THArgCheck(kernel->size[0] == input->size[0] , 2, "conv2DRevger : Input batch and kernel batch is not same size");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize4d)(r_,nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    /*THZTensor_(zero)(r_);*/

#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }
  }
  else if (beta != 1)
  {
    /*THZTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] *= beta;
    }
  }

#pragma omp parallel for private(k)
  for(k = 0; k < nKernelPlane; k++)
  {
    long i;
    for(i = 0; i < nInputPlane; i++)
    {
      long p;
      for(p = 0; p < nbatch; p++)
      {
        /* get kernel */
        real *ptr_weight = weight_data + p*kstride0 + k*kstride1;
        /* get output */
        real *ptr_output = output_data + k*nInputPlane*nOutputCols*nOutputRows + i*nOutputCols*nOutputRows;
        /* get input */
        real *ptr_input = input_data + p*istride0 + i*istride1;

        /* do image, kernel convolution */
        THZTensor_(validXCorr2DRevptr)(ptr_output,
                                      alpha,
                                      ptr_input,  nInputRows,  nInputCols,
                                      ptr_weight, nKernelRows, nKernelCols,
                                      srow, scol);
        /* Next output plane */
        /* output_data += nOutputCols*nOutputRows; */
      }
    }
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}


/*
  3D input, 3D kernel, 4D output
  like rank1 update
  A <- xx' + beta*A
*/
void THZTensor_(conv2Dger)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelPlane, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0;

  THZTensor *input;
  THZTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can 'X' or 'C'");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0 = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];
  nOutputPlane = nInputPlane * kernel->size[0];

  THArgCheck((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dger : Input image is smaller than kernel");

  if (*vf == 'F') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { /* valid */
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize4d)(r_, nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    /*THZTensor_(zero)(r_);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }
  }
  else if (beta != 1)
  {
    /*THZTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] *= beta;
    }
  }

#pragma omp parallel for private(k)
  for(k = 0; k < nKernelPlane; k++)
  {
    long i;
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get output */
      real *ptr_output = output_data + k*nInputPlane*nOutputCols*nOutputRows + i*nOutputCols*nOutputRows;
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      if (*vf == 'F')
        if (*xc == 'X')
          THZTensor_(fullXCorr2Dptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
        else
          THZTensor_(fullConv2Dptr)(ptr_output,
                                   alpha,
                                   ptr_input,  nInputRows,  nInputCols,
                                   ptr_weight, nKernelRows, nKernelCols,
                                   srow, scol);
      else
        if (*xc == 'X')
          THZTensor_(validXCorr2Dptr)(ptr_output,
                                     alpha,
                                     ptr_input,  nInputRows,  nInputCols,
                                     ptr_weight, nKernelRows, nKernelCols,
                                     srow, scol);
        else
          THZTensor_(validConv2Dptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows; */
    }
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}


/*
  3D input, 4D kernel, 3D output
  matrix vector product like
  y <- Ax + beta*y
*/
void THZTensor_(conv2Dmv)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0, kstride1;
  THZTensor *input;
  THZTensor* kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can 'X' or 'C'");

  input = THZTensor_(newContiguous)(t_);
  if (!(k_->stride[3] == 1) || !(k_->stride[2] == k_->size[3])) {
    kernel = THZTensor_(newContiguous)(k_);
  } else {
    THZTensor_(retain)(k_);
    kernel = k_;
  }

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0    = kernel->stride[0];
  kstride1    = kernel->stride[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dmv : Input image is smaller than kernel");

  if (*vf == 'F') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { /* valid */
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize3d)(r_, nOutputPlane, nOutputRows, nOutputCols);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    /*THZTensor_(zero)(r_);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }
  }
  else if (beta != 1)
  {
    /*THZTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] *= beta;
    }
  }

#pragma omp parallel for private(k)
  for(k = 0; k < nOutputPlane; k++)
  {
    long i;
    /* get output */
    real *ptr_output = output_data + k*nOutputCols*nOutputRows;
    for(i = 0; i < nInputPlane; i++)
    {
      /* get kernel */
      real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
      /* get input */
      real *ptr_input = input_data + i*istride0;

      /* do image, kernel convolution */
      if (*vf == 'F')
        if (*xc == 'X')
          THZTensor_(fullXCorr2Dptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
        else
          THZTensor_(fullConv2Dptr)(ptr_output,
                                   alpha,
                                   ptr_input,  nInputRows,  nInputCols,
                                   ptr_weight, nKernelRows, nKernelCols,
                                   srow, scol);
      else
        if (*xc == 'X')
          THZTensor_(validXCorr2Dptr)(ptr_output,
                                     alpha,
                                     ptr_input,  nInputRows,  nInputCols,
                                     ptr_weight, nKernelRows, nKernelCols,
                                     srow, scol);
        else
          THZTensor_(validConv2Dptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
    }
    /* Next output plane */
    /* output_data += nOutputCols*nOutputRows;*/
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}


/*
  3D input, 4D kernel, 3D output
  matrix vector product like
  y <- Ax + beta*y
*/
void THZTensor_(conv2Dmm)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long kstride0, kstride1;
  THZTensor *input;
  THZTensor* kernel;
  long nbatch;
  long nelem;
  real *input_data;
  real *weight_data;
  real *output_data;
  long p;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can 'X' or 'C'");

  input = THZTensor_(newContiguous)(t_);
  if (!(k_->stride[3] == 1) || !(k_->stride[2] == k_->size[3])) {
    kernel = THZTensor_(newContiguous)(k_);
  } else {
    THZTensor_(retain)(k_);
    kernel = k_;
  }

  nbatch = input->size[0];
  nInputPlane = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0    = kernel->stride[0];
  kstride1    = kernel->stride[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dmv : Input image is smaller than kernel");

  if (*vf == 'F') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { /* valid */
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize4d)(r_, nbatch, nOutputPlane, nOutputRows, nOutputCols);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    /*THZTensor_(zero)(r_);*/
#pragma omp parallel for private(p)
    for (p=0; p < r_->size[0]; p++)
    {
      long k;
      for (k = 0; k < r_->size[1]; k++)
      {
        real* ptr_output = output_data + p*nOutputPlane*nOutputRows*nOutputCols + k*nOutputCols*nOutputRows;
        long l;
        for (l = 0; l < nOutputRows*nOutputCols; l++)
          ptr_output[l] = 0.0;
      }
    }
  }
  else if (beta != 1)
  {
    /*THZTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(p)
    for(p=0; p < r_->size[0]; p++)
    {
      long k;
      for (k = 0; k < r_->size[1]; k++)
      {
        real* ptr_output = output_data + p*nOutputPlane*nOutputRows*nOutputCols + k*nOutputCols*nOutputRows;
        long l;
        for (l = 0; l < nOutputRows*nOutputCols; l++)
          ptr_output[l] *= beta;
      }
    }
  }

#pragma omp parallel for private(p)
  for(p=0; p < nbatch; p++)
  {
    long k;
    for(k = 0; k < nOutputPlane; k++)
    {
      long i;
      /* get output */
      real *ptr_output = output_data + p*nOutputPlane*nOutputCols*nOutputRows + k*nOutputCols*nOutputRows;
      for(i = 0; i < nInputPlane; i++)
      {
        /* get kernel */
        real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
        /* get input */
        real *ptr_input = input_data + p*nInputPlane*nInputRows*nInputCols + i*nInputRows*nInputCols;

        /* do image, kernel convolution */
        if (*vf == 'F')
          if (*xc == 'X')
            THZTensor_(fullXCorr2Dptr)(ptr_output,
                                      alpha,
                                      ptr_input,  nInputRows,  nInputCols,
                                      ptr_weight, nKernelRows, nKernelCols,
                                      srow, scol);
          else
            THZTensor_(fullConv2Dptr)(ptr_output,
                                     alpha,
                                     ptr_input,  nInputRows,  nInputCols,
                                     ptr_weight, nKernelRows, nKernelCols,
                                     srow, scol);
        else
          if (*xc == 'X')
            THZTensor_(validXCorr2Dptr)(ptr_output,
                                       alpha,
                                       ptr_input,  nInputRows,  nInputCols,
                                       ptr_weight, nKernelRows, nKernelCols,
                                       srow, scol);
          else
            THZTensor_(validConv2Dptr)(ptr_output,
                                      alpha,
                                      ptr_input,  nInputRows,  nInputCols,
                                      ptr_weight, nKernelRows, nKernelCols,
                                      srow, scol);
      }
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows;*/
    }
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}


/*
  2D input, 2D kernel, 2D output
  scalar multiplication like
  y <- x*y + beta*y
*/
void THZTensor_(conv2Dmul)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc)
{
  THZTensor *input;
  THZTensor* kernel;
  long nInputRows;
  long nInputCols;
  long nKernelRows;
  long nKernelCols;
  long nOutputRows, nOutputCols;
  real *ptr_input;
  real *ptr_weight;
  real *output_data;
  long nelem;

  THArgCheck(t_->nDimension == 2 , 3, "input: 2D Tensor expected");
  THArgCheck(k_->nDimension == 2 , 4, "kernel: 2D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  nInputRows  = input->size[0];
  nInputCols  = input->size[1];
  nKernelRows = kernel->size[0];
  nKernelCols = kernel->size[1];

  THArgCheck((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dmul : Input image is smaller than kernel");

  nOutputRows = THZTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THZTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize2d)(r_, nOutputRows, nOutputCols);
  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
    THZTensor_(zero)(r_);
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  ptr_input = THZTensor_(data)(input);
  ptr_weight = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);


  /* do image, kernel convolution */
  THZTensor_(conv2d)(output_data,
                    alpha,
                    ptr_input, nInputRows, nInputCols,
                    ptr_weight, nKernelRows, nKernelCols,
                    srow, scol, vf, xc);
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}

/*
  3D input, 3D kernel, 3D output
  component wise multiplication like
  y <- y.*x + beta*y
*/
void THZTensor_(conv2Dcmul)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0;
  THZTensor *input;
  THZTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0    = kernel->stride[0];
  nOutputPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck(nOutputPlane == nInputPlane, 2, "invalid number of input/kernel planes");
  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dcmul : Input image is smaller than kernel");

  nOutputRows = THZTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THZTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize3d)(r_, nOutputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    THZTensor_(zero)(r_);
  }
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  for(k = 0; k < nOutputPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data + k*kstride0;
    /* get input */
    real *ptr_input = input_data + k*istride0;

    /* do image, kernel convolution */
    THZTensor_(conv2d)(output_data,
                      alpha,
                      ptr_input, nInputRows, nInputCols,
                      ptr_weight, nKernelRows, nKernelCols,
                      srow, scol, vf, xc);
    /* Next output plane */
    output_data += nOutputCols*nOutputRows;
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}

/*
  3D input, 3D kernel, 3D output
  component wise multiplication like with a permutation map
  y <- y.*x + beta*y
*/
void THZTensor_(conv2Dmap)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, THZTensor *map, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;
  long istride0, kstride0;
  THZTensor *input;
  THZTensor* kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nmaps;
  long nelem;
  long k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(map->nDimension == 2 , 4, "map: 2D Tensor expected");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0    = kernel->stride[0];
  nOutputPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck(nOutputPlane == nInputPlane, 2, "invalid number of input/kernel planes");
  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols)
              || *vf == 'F', 2, "conv2Dmap : Input image is smaller than kernel");

  nOutputRows = THZTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THZTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize3d)(r_, nOutputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    THZTensor_(zero)(r_);
  }
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  nmaps = map->size[0];

  for(k = 0; k < nmaps; k++)
  {
    /* get indices */
    long from = (long)THZTensor_(get2d)(map,k,0)-1;
    long to   = (long)THZTensor_(get2d)(map,k,1)-1;

    /* get kernel */
    real *ptr_weight = weight_data + k*kstride0;
    /* get input */
    real *ptr_input = input_data + from*istride0;
    /* get output */
    real *ptr_output = output_data + to*nOutputRows*nOutputCols;

    /* do image, kernel convolution */
    THZTensor_(conv2d)(ptr_output,
                      alpha,
                      ptr_input, nInputRows, nInputCols,
                      ptr_weight, nKernelRows, nKernelCols,
                      srow, scol, vf, xc);
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}

/*
  4D input, 4D kernel, 5D output
  like rank1 update
  A <- xx' + beta*A
  for sr,sc=1 this is equivalent to xcorr2Dger, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
void THZTensor_(conv3DRevger)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_,
                             long sdepth, long srow, long scol)
{
  long nInputPlane, nInputDepth, nInputRows, nInputCols;
  long nKernelPlane, nKernelDepth, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
  long istride0, kstride0;
  THZTensor *input;
  THZTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k, i;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(sdepth >= 1, 5, "Stride should be a positive integer");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0 = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelDepth= kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];
  nOutputPlane = nInputPlane * kernel->size[0];

  THArgCheck(nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "conv3DRevger : Input image is smaller than kernel");

  nOutputDepth = nInputDepth - (nKernelDepth - 1) * sdepth;
  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize5d)(r_,nKernelPlane, nInputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    THZTensor_(zero)(r_);
  }
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  for(k = 0; k < nKernelPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      THZTensor_(validXCorr3DRevptr)(output_data,
                                    alpha,
                                    ptr_input,  nInputDepth, nInputRows,  nInputCols,
                                    ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                    sdepth, srow, scol);
      /* Next output plane */
      output_data += nOutputDepth*nOutputCols*nOutputRows;
    }
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}


/*
  4D input, 4D kernel, 5D output
  like rank1 update
  A <- xx' + beta*A
*/
void THZTensor_(conv3Dger)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_,
                          long sdepth, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputDepth, nInputRows, nInputCols;
  long nKernelPlane, nKernelDepth, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
  long istride0, kstride0;
  THZTensor *input;
  THZTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k, i;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(sdepth >= 1, 5, "Stride should be a positive integer");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 8, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 8, "type of convolution can 'X' or 'C'");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0     = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelDepth = kernel->size[1];
  nKernelRows  = kernel->size[2];
  nKernelCols  = kernel->size[3];
  nOutputPlane = nInputPlane * kernel->size[0];

  THArgCheck((nInputDepth >= nKernelDepth
              && nInputRows >= nKernelRows
              && nInputCols >= nKernelCols)
             || *vf == 'F', 2, "conv3Dger : Input image is smaller than kernel");

  nOutputDepth = THZTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THZTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THZTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize5d)(r_,nKernelPlane, nInputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    THZTensor_(zero)(r_);
  }
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  for(k = 0; k < nKernelPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      THZTensor_(conv3d)(output_data,
                        alpha,
                        ptr_input,  nInputDepth, nInputRows,  nInputCols,
                        ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                        sdepth, srow, scol, vf, xc);

      /* Next output plane */
      output_data += nOutputDepth*nOutputCols*nOutputRows;
    }
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}

/*
  4D input, 5D kernel, 4D output
  matrix vector product like
  y <- Ax + beta*y
*/
void THZTensor_(conv3Dmv)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_,
                         long sdepth, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputDepth, nInputRows, nInputCols;
  long nKernelDepth, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
  long istride0, kstride0, kstride1;
  THZTensor *input;
  THZTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k, i;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 5 , 4, "kernel: 5D Tensor expected");
  THArgCheck(sdepth >= 1, 5, "Stride should be a positive integer");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 8, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 8, "type of convolution can 'X' or 'C'");

  input = THZTensor_(newContiguous)(t_);
  if (!(k_->stride[4] == 1) || !(k_->stride[3] == k_->size[4])) {
    kernel = THZTensor_(newContiguous)(k_);
  } else {
    THZTensor_(retain)(k_);
    kernel = k_;
  }

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0    = kernel->stride[0];
  kstride1    = kernel->stride[1];
  nKernelDepth = kernel->size[2];
  nKernelRows = kernel->size[3];
  nKernelCols = kernel->size[4];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv3Dmv : Input image is smaller than kernel");

  nOutputDepth = THZTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THZTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THZTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize4d)(r_, nOutputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    THZTensor_(zero)(r_);
  }
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  for(k = 0; k < nOutputPlane; k++)
  {
    for(i = 0; i < nInputPlane; i++)
    {
      /* get kernel */
      real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
      /* get input */
      real *ptr_input = input_data + i*istride0;

      /* do image, kernel convolution */
      THZTensor_(conv3d)(output_data,
                        alpha,
                        ptr_input,  nInputDepth, nInputRows,  nInputCols,
                        ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                        sdepth, srow, scol, vf, xc);
    }
    /* Next output plane */
    output_data += nOutputDepth*nOutputCols*nOutputRows;
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}

/*
  3D input, 3D kernel, 3D output
  scalar multiplication like
  y <- x*y + beta*y
*/
void THZTensor_(conv3Dmul)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_,
                          long sdepth, long srow, long scol, const char *vf, const char *xc)
{
  THZTensor *input;
  THZTensor* kernel;
  long nInputDepth;
  long nInputRows;
  long nInputCols;
  long nKernelDepth;
  long nKernelRows;
  long nKernelCols;
  long nOutputDepth, nOutputRows, nOutputCols;
  real *ptr_input;
  real *ptr_weight;
  real *output_data;
  long nelem;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(sdepth >= 1, 5, "Stride should be a positive integer");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 8, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 8, "type of convolution can 'X' or 'C'");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  nInputDepth = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];
  nKernelDepth = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv3Dmul : Input image is smaller than kernel");

  nOutputDepth = THZTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THZTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THZTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize3d)(r_, nOutputDepth, nOutputRows, nOutputCols);
  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
    THZTensor_(zero)(r_);
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  ptr_input = THZTensor_(data)(input);
  ptr_weight = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);


  /* do image, kernel convolution */
  THZTensor_(conv3d)(output_data,
                    alpha,
                    ptr_input,  nInputDepth, nInputRows,  nInputCols,
                    ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                    sdepth, srow, scol, vf, xc);
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}

/*
  4D input, 4D kernel, 4D output
  component wise multiplication like
  y <- y.*x + beta*y
*/
void THZTensor_(conv3Dcmul)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_,
                           long sdepth, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputDepth, nInputRows, nInputCols;
  long nKernelDepth, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
  long istride0, kstride0;

  THZTensor *input;
  THZTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nelem;
  long k;

  THArgCheck(t_->nDimension == 4 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can 'X' or 'C'");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  nInputPlane = input->size[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0    = kernel->stride[0];
  nOutputPlane = kernel->size[0];
  nKernelDepth = kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nOutputPlane == nInputPlane, 2, "invalid number of input/kernel planes");
  THArgCheck( (nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv3Dcmul : Input image is smaller than kernel");

  nOutputDepth = THZTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THZTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THZTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize4d)(r_, nOutputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    THZTensor_(zero)(r_);
  }
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  for(k = 0; k < nOutputPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data + k*kstride0;
    /* get input */
    real *ptr_input = input_data + k*istride0;

    /* do image, kernel convolution */
    THZTensor_(conv3d)(output_data,
                      alpha,
                      ptr_input,  nInputDepth, nInputRows,  nInputCols,
                      ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                      sdepth, srow, scol, vf, xc);

    /* Next output plane */
    output_data += nOutputDepth*nOutputCols*nOutputRows;
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}

/*
  4D input, 4D kernel, 4D output
  component wise multiplication like with a permutation map
  y <- y.*x + beta*y
*/
void THZTensor_(conv3Dmap)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, THZTensor *map,
                          long sdepth, long srow, long scol, const char *vf, const char *xc)
{
  long nInputPlane, nInputDepth, nInputRows, nInputCols;
  long nKernelDepth, nKernelRows, nKernelCols;
  long nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
  long istride0, kstride0;

  THZTensor *input;
  THZTensor *kernel;
  long nelem;
  real *input_data;
  real *weight_data;
  real *output_data;
  long nmaps;
  long k;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(map->nDimension == 2 , 4, "map: 2D Tensor expected");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 8, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 8, "type of convolution can 'X' or 'C'");

  input = THZTensor_(newContiguous)(t_);
  kernel = THZTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  nInputPlane = input->size[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0    = kernel->stride[0];
  nOutputPlane = kernel->size[0];
  nKernelDepth = kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nOutputPlane == nInputPlane, 2, "invalid number of input/kernel planes");
  THArgCheck((nInputDepth >= nKernelDepth
              && nInputRows >= nKernelRows
              && nInputCols >= nKernelCols) || *vf == 'F',
             2, "conv3Dmap : Input image is smaller than kernel");

  nOutputDepth = THZTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THZTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THZTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THZTensor_(nElement)(r_);
  THZTensor_(resize4d)(r_, nOutputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THZTensor_(nElement)(r_))
  {
    THZTensor_(zero)(r_);
  }
  else if (beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  input_data = THZTensor_(data)(input);
  weight_data = THZTensor_(data)(kernel);
  output_data = THZTensor_(data)(r_);

  nmaps = map->size[0];

  for(k = 0; k < nmaps; k++)
  {
    /* get indices */
    long from = (long)THZTensor_(get2d)(map,k,0)-1;
    long to   = (long)THZTensor_(get2d)(map,k,1)-1;

    /* get kernel */
    real *ptr_weight = weight_data + k*kstride0;
    /* get input */
    real *ptr_input = input_data + from*istride0;
    /* get output */
    real *ptr_output = output_data + to*nOutputDepth*nOutputRows*nOutputCols;

    /* do image, kernel convolution */
    THZTensor_(conv3d)(ptr_output,
                      alpha,
                      ptr_input,  nInputDepth, nInputRows,  nInputCols,
                      ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                      sdepth, srow, scol, vf, xc);
  }
  THZTensor_(free)(input);
  THZTensor_(free)(kernel);
}

#endif
