#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "thermodynamic_functions.h"
#include "entropies.h"

void second_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

    const ssize_t imax = dims->nlg[0]-1;
    const ssize_t jmax = dims->nlg[1]-1;
    const ssize_t kmax = dims->nlg[2]-1;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1]) * velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else

    return;
}

void fourth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 1;
    const ssize_t jmin = 1;
    const ssize_t kmin = 1;

    const ssize_t imax = dims->nlg[0]-2;
    const ssize_t jmax = dims->nlg[1]-2;
    const ssize_t kmax = dims->nlg[2]-2;


    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sm1 = -sp1 ;


    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } //end if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else
    return;
}



void sixth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 2;
    const ssize_t jmin = 2;
    const ssize_t kmin = 2;

    const ssize_t imax = dims->nlg[0]-3;
    const ssize_t jmax = dims->nlg[1]-3;
    const ssize_t kmax = dims->nlg[2]-3;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_6(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3])*velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_6(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void eighth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 3;
    const ssize_t jmin = 3;
    const ssize_t kmin = 3;

    const ssize_t imax = dims->nlg[0]-4;
    const ssize_t jmax = dims->nlg[1]-4;
    const ssize_t kmax = dims->nlg[2]-4;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_8(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4])*velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_8(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    return;
}

void upwind_first_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 1;
    const ssize_t jmin = 1;
    const ssize_t kmin = 1;

    const ssize_t imax = dims->nlg[0]-1;
    const ssize_t jmax = dims->nlg[1]-1;
    const ssize_t kmax = dims->nlg[2]-1;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = scalar[ijk];
                    // Up wind for negative velocity
                    const double phim =scalar[ijk+sp1];
                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = scalar[ijk];
                    // Up wind for negative velocity
                    const double phim =scalar[ijk+sp1];
                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void weno_third_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 1;
    const ssize_t jmin = 1;
    const ssize_t kmin = 1;

    const ssize_t imax = dims->nlg[0]-2;
    const ssize_t jmax = dims->nlg[1]-2;
    const ssize_t kmax = dims->nlg[2]-2;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sm1 = -sp1 ;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno3(scalar[ijk+sm1],
                                                     scalar[ijk],
                                                     scalar[ijk+sp1]);

                    // Up wind for negative velocity
                    const double phim = interp_weno3(scalar[ijk+sp2],
                                                     scalar[ijk+sp1],
                                                     scalar[ijk]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno3(scalar[ijk+sm1],
                                                     scalar[ijk],
                                                     scalar[ijk+sp1]);

                    // Up wind for negative velocity
                    const double phim = interp_weno3(scalar[ijk+sp2],
                                                     scalar[ijk+sp1],
                                                     scalar[ijk]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}


void weno_fifth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 2;
    const ssize_t jmin = 2;
    const ssize_t kmin = 2;

    const ssize_t imax = dims->nlg[0]-3;
    const ssize_t jmax = dims->nlg[1]-3;
    const ssize_t kmax = dims->nlg[2]-3;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2]);

                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2]);

                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void weno_seventh_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 3;
    const ssize_t jmin = 3;
    const ssize_t kmin = 3;

    const ssize_t imax = dims->nlg[0]-4;
    const ssize_t jmax = dims->nlg[1]-4;
    const ssize_t kmax = dims->nlg[2]-4;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno7(scalar[ijk + sm3],
                                                     scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp3]);

                    // Up wind for negative velocity
                    const double phim = interp_weno7(scalar[ijk + sp4],
                                                     scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk + sm2]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                     //Upwind for positive velocity
                    const double phip = interp_weno7(scalar[ijk + sm3],
                                                     scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp3]);

                    // Up wind for negative velocity
                    const double phim = interp_weno7(scalar[ijk + sp4],
                                                     scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk + sm2]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    return;
}

void weno_ninth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 4;
    const ssize_t jmin = 4;
    const ssize_t kmin = 4;

    const ssize_t imax = dims->nlg[0]-5;
    const ssize_t jmax = dims->nlg[1]-5;
    const ssize_t kmax = dims->nlg[2]-5;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sp5 = 5 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;
    const ssize_t sm4 = -4*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno9(scalar[ijk + sm4],
                                                     scalar[ijk + sm3],
                                                     scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp3],
                                                     scalar[ijk + sp4]);

                    // Up wind for negative velocity
                    const double phim = interp_weno9(scalar[ijk + sp5],
                                                     scalar[ijk + sp4],
                                                     scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk + sm2],
                                                     scalar[ijk + sm3]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno9(scalar[ijk + sm4],
                                                     scalar[ijk + sm3],
                                                     scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp3],
                                                     scalar[ijk + sp4]);

                    // Up wind for negative velocity
                    const double phim = interp_weno9(scalar[ijk + sp5],
                                                     scalar[ijk + sp4],
                                                     scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk + sm2],
                                                     scalar[ijk + sm3]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void weno_eleventh_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 5;
    const ssize_t jmin = 5;
    const ssize_t kmin = 5;

    const ssize_t imax = dims->nlg[0]-6;
    const ssize_t jmax = dims->nlg[1]-6;
    const ssize_t kmax = dims->nlg[2]-6;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sp5 = 5 * sp1;
    const ssize_t sp6 = 6 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;
    const ssize_t sm4 = -4*sp1;
    const ssize_t sm5 = -5*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno11(scalar[ijk + sm5],
                                                      scalar[ijk + sm4],
                                                      scalar[ijk + sm3],
                                                      scalar[ijk + sm2],
                                                      scalar[ijk + sm1],
                                                      scalar[ijk],
                                                      scalar[ijk + sp1],
                                                      scalar[ijk + sp2],
                                                      scalar[ijk + sp3],
                                                      scalar[ijk + sp4],
                                                      scalar[ijk + sp5]);

                    // Up wind for negative velocity
                    const double phim = interp_weno11(scalar[ijk + sp6],
                                                      scalar[ijk + sp5],
                                                      scalar[ijk + sp4],
                                                      scalar[ijk + sp3],
                                                      scalar[ijk + sp2],
                                                      scalar[ijk + sp1],
                                                      scalar[ijk],
                                                      scalar[ijk + sm1],
                                                      scalar[ijk + sm2],
                                                      scalar[ijk + sm3],
                                                      scalar[ijk + sm4]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno11(scalar[ijk + sm5],
                                                      scalar[ijk + sm4],
                                                      scalar[ijk + sm3],
                                                      scalar[ijk + sm2],
                                                      scalar[ijk + sm1],
                                                      scalar[ijk],
                                                      scalar[ijk + sp1],
                                                      scalar[ijk + sp2],
                                                      scalar[ijk + sp3],
                                                      scalar[ijk + sp4],
                                                      scalar[ijk + sp5]);

                    // Up wind for negative velocity
                    const double phim = interp_weno11(scalar[ijk + sp6],
                                                      scalar[ijk + sp5],
                                                      scalar[ijk + sp4],
                                                      scalar[ijk + sp3],
                                                      scalar[ijk + sp2],
                                                      scalar[ijk + sp1],
                                                      scalar[ijk],
                                                      scalar[ijk + sm1],
                                                      scalar[ijk + sm2],
                                                      scalar[ijk + sm3],
                                                      scalar[ijk + sm4]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}


// This assumes velocity contains velocities at cell centers, not interfaces
void hiweno_third_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 1;
    const ssize_t jmin = 1;
    const ssize_t kmin = 1;

    const ssize_t imax = dims->nlg[0]-2;
    const ssize_t jmax = dims->nlg[1]-2;
    const ssize_t kmax = dims->nlg[2]-2;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sm1 = -sp1 ;
    
    double a;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno3(velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1]);

                    // Up wind for negative velocity
                    const double phim = interp_weno3(velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k]);
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  (a >= 0)*phip + (a < 0)*phim;
                } // End k loop 
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno3(velocity[ijk+sm1]*scalar[ijk+sm1],
                                                     velocity[ijk]*scalar[ijk],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]);

                    // Up wind for negative velocity
                    const double phim = interp_weno3(velocity[ijk+sp2]*scalar[ijk+sp2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1],
                                                     velocity[ijk]*scalar[ijk]);
                                                     
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  ((a >= 0)*phip + (a < 0)*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void hiweno_fifth_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 2;
    const ssize_t jmin = 2;
    const ssize_t kmin = 2;

    const ssize_t imax = dims->nlg[0]-3;
    const ssize_t jmax = dims->nlg[1]-3;
    const ssize_t kmax = dims->nlg[2]-3;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;

    double a;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno5(velocity[ijk+sm2]*scalar[ijk+sm2]*rho0_half[k-2],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2]);

                    // Up wind for negative velocity
                    const double phim = interp_weno5(velocity[ijk+sp3]*scalar[ijk+sp3]*rho0_half[k+3],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1]);
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  (a >= 0)*phip + (a < 0)*phim;
                } // End k loop 
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno5(velocity[ijk+sm2]*scalar[ijk+sm2],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1],
                                                     velocity[ijk]*scalar[ijk],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]);

                    // Up wind for negative velocity
                    const double phim = interp_weno5(velocity[ijk+sp3]*scalar[ijk+sp3],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1],
                                                     velocity[ijk]*scalar[ijk],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]);
                                                     
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  ((a >= 0)*phip + (a < 0)*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void hiweno_seventh_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 3;
    const ssize_t jmin = 3;
    const ssize_t kmin = 3;

    const ssize_t imax = dims->nlg[0]-4;
    const ssize_t jmax = dims->nlg[1]-4;
    const ssize_t kmax = dims->nlg[2]-4;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;

    double a;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno7(velocity[ijk+sm3]*scalar[ijk+sm3]*rho0_half[k-3],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2]*rho0_half[k-2],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3]*rho0_half[k+3]);

                    // Up wind for negative velocity
                    const double phim = interp_weno7(velocity[ijk+sp4]*scalar[ijk+sp4]*rho0_half[k+4],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3]*rho0_half[k+3],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2]*rho0_half[k-2]);
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  (a >= 0)*phip + (a < 0)*phim;
                } // End k loop 
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno7(velocity[ijk+sm3]*scalar[ijk+sm3],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1],
                                                     velocity[ijk]*scalar[ijk],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3]);

                    // Up wind for negative velocity
                    const double phim = interp_weno7(velocity[ijk+sp4]*scalar[ijk+sp4],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1],
                                                     velocity[ijk]*scalar[ijk],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2]);
                                                     
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  ((a >= 0)*phip + (a < 0)*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void hiweno_ninth_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 4;
    const ssize_t jmin = 4;
    const ssize_t kmin = 4;

    const ssize_t imax = dims->nlg[0]-5;
    const ssize_t jmax = dims->nlg[1]-5;
    const ssize_t kmax = dims->nlg[2]-5;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sp5 = 5 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;
    const ssize_t sm4 = -4*sp1;

    double a;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno9(velocity[ijk+sm4]*scalar[ijk+sm4]*rho0_half[k-4],
                                                     velocity[ijk+sm3]*scalar[ijk+sm3]*rho0_half[k-3],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2]*rho0_half[k-2],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3]*rho0_half[k+3],
                                                     velocity[ijk+sp4]*scalar[ijk+sp4]*rho0_half[k+4]);

                    // Up wind for negative velocity
                    const double phim = interp_weno9(velocity[ijk+sp5]*scalar[ijk+sp5]*rho0_half[k+5],
                                                     velocity[ijk+sp4]*scalar[ijk+sp4]*rho0_half[k+4],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3]*rho0_half[k+3],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2]*rho0_half[k-2],
                                                     velocity[ijk+sm3]*scalar[ijk+sm3]*rho0_half[k-3]);
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  (a >= 0)*phip + (a < 0)*phim;
                } // End k loop 
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno9(velocity[ijk+sm4]*scalar[ijk+sm4],
                                                     velocity[ijk+sm3]*scalar[ijk+sm3],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1],
                                                     velocity[ijk]*scalar[ijk],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3],
                                                     velocity[ijk+sp4]*scalar[ijk+sp4]);

                    // Up wind for negative velocity
                    const double phim = interp_weno9(velocity[ijk+sp5]*scalar[ijk+sp5],
                                                     velocity[ijk+sp4]*scalar[ijk+sp4],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1],
                                                     velocity[ijk]*scalar[ijk],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2],
                                                     velocity[ijk+sm3]*scalar[ijk+sm3]);
                                                     
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  ((a >= 0)*phip + (a < 0)*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void hiweno_eleventh_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 5;
    const ssize_t jmin = 5;
    const ssize_t kmin = 5;

    const ssize_t imax = dims->nlg[0]-6;
    const ssize_t jmax = dims->nlg[1]-6;
    const ssize_t kmax = dims->nlg[2]-6;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sp5 = 5 * sp1;
    const ssize_t sp6 = 6 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;
    const ssize_t sm4 = -4*sp1;
    const ssize_t sm5 = -5*sp1;

    double a;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno11(velocity[ijk+sm5]*scalar[ijk+sm5]*rho0_half[k-5],
                                                     velocity[ijk+sm4]*scalar[ijk+sm4]*rho0_half[k-4],
                                                     velocity[ijk+sm3]*scalar[ijk+sm3]*rho0_half[k-3],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2]*rho0_half[k-2],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3]*rho0_half[k+3],
                                                     velocity[ijk+sp4]*scalar[ijk+sp4]*rho0_half[k+4],
                                                     velocity[ijk+sp5]*scalar[ijk+sp5]*rho0_half[k+5]);

                    // Up wind for negative velocity
                    const double phim = interp_weno11(velocity[ijk+sp6]*scalar[ijk+sp6]*rho0_half[k+6],
                                                     velocity[ijk+sp5]*scalar[ijk+sp5]*rho0_half[k+5],
                                                     velocity[ijk+sp4]*scalar[ijk+sp4]*rho0_half[k+4],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3]*rho0_half[k+3],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2]*rho0_half[k+2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1]*rho0_half[k+1],
                                                     velocity[ijk]*scalar[ijk]*rho0_half[k],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1]*rho0_half[k-1],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2]*rho0_half[k-2],
                                                     velocity[ijk+sm3]*scalar[ijk+sm3]*rho0_half[k-3],
                                                     velocity[ijk+sm4]*scalar[ijk+sm4]*rho0_half[k-4]);
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  (a >= 0)*phip + (a < 0)*phim;
                } // End k loop 
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno11(velocity[ijk+sm5]*scalar[ijk+sm5],
                                                      velocity[ijk+sm4]*scalar[ijk+sm4],
                                                      velocity[ijk+sm3]*scalar[ijk+sm3],
                                                      velocity[ijk+sm2]*scalar[ijk+sm2],
                                                      velocity[ijk+sm1]*scalar[ijk+sm1],
                                                      velocity[ijk]*scalar[ijk],
                                                      velocity[ijk+sp1]*scalar[ijk+sp1],
                                                      velocity[ijk+sp2]*scalar[ijk+sp2],
                                                      velocity[ijk+sp3]*scalar[ijk+sp3],
                                                      velocity[ijk+sp4]*scalar[ijk+sp4],
                                                      velocity[ijk+sp5]*scalar[ijk+sp5]);

                    // Up wind for negative velocity
                    const double phim = interp_weno11(velocity[ijk+sp6]*scalar[ijk+sp6],
                                                     velocity[ijk+sp5]*scalar[ijk+sp5],
                                                     velocity[ijk+sp4]*scalar[ijk+sp4],
                                                     velocity[ijk+sp3]*scalar[ijk+sp3],
                                                     velocity[ijk+sp2]*scalar[ijk+sp2],
                                                     velocity[ijk+sp1]*scalar[ijk+sp1],
                                                     velocity[ijk]*scalar[ijk],
                                                     velocity[ijk+sm1]*scalar[ijk+sm1],
                                                     velocity[ijk+sm2]*scalar[ijk+sm2],
                                                     velocity[ijk+sm3]*scalar[ijk+sm3],
                                                     velocity[ijk+sm4]*scalar[ijk+sm4]);
                                                     
                     
                    a = velocity[ijk] + velocity[ijk+sp1]; // for upwinding
                    flux[ijk] =  ((a >= 0)*phip + (a < 0)*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void hiweno_third_order_nonconserv(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 1;
    const ssize_t jmin = 1;
    const ssize_t kmin = 1;

    const ssize_t imax = dims->nlg[0]-2;
    const ssize_t jmax = dims->nlg[1]-2;
    const ssize_t kmax = dims->nlg[2]-2;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sm1 = -sp1 ;

    
    // The non-conservative form doesn't require rho or alpha at all - so there's no need to distinguish cases :)
    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;
                
                //Upwind for positive velocity
                const double phip = interp_weno3(scalar[ijk+sm1],
                                                 scalar[ijk],
                                                 scalar[ijk+sp1]);

                // Up wind for negative velocity
                const double phim = interp_weno3(scalar[ijk+sp2],
                                                 scalar[ijk+sp1],
                                                 scalar[ijk]);

                flux[ijk] = (velocity[ijk + sp1] >= 0)*phip + (velocity[ijk + sp1] < 0)*phim;
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}

void hiweno_fifth_order_nonconserv(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 2;
    const ssize_t jmin = 2;
    const ssize_t kmin = 2;

    const ssize_t imax = dims->nlg[0]-3;
    const ssize_t jmax = dims->nlg[1]-3;
    const ssize_t kmax = dims->nlg[2]-3;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;

                //Upwind for positive velocity
                const double phip = interp_weno5(scalar[ijk + sm2],
                                                 scalar[ijk + sm1],
                                                 scalar[ijk],
                                                 scalar[ijk + sp1],
                                                 scalar[ijk + sp2]);

                // Up wind for negative velocity
                const double phim = interp_weno5(scalar[ijk + sp3],
                                                 scalar[ijk + sp2],
                                                 scalar[ijk + sp1],
                                                 scalar[ijk],
                                                 scalar[ijk + sm1]);

                flux[ijk] = (velocity[ijk + sp1] >= 0)*phip + (velocity[ijk + sp1] < 0)*phim;
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}

void hiweno_seventh_order_nonconserv(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 3;
    const ssize_t jmin = 3;
    const ssize_t kmin = 3;

    const ssize_t imax = dims->nlg[0]-4;
    const ssize_t jmax = dims->nlg[1]-4;
    const ssize_t kmax = dims->nlg[2]-4;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;


    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;

                 //Upwind for positive velocity
                const double phip = interp_weno7(scalar[ijk + sm3],
                                                 scalar[ijk + sm2],
                                                 scalar[ijk + sm1],
                                                 scalar[ijk],
                                                 scalar[ijk + sp1],
                                                 scalar[ijk + sp2],
                                                 scalar[ijk + sp3]);

                // Up wind for negative velocity
                const double phim = interp_weno7(scalar[ijk + sp4],
                                                 scalar[ijk + sp3],
                                                 scalar[ijk + sp2],
                                                 scalar[ijk + sp1],
                                                 scalar[ijk],
                                                 scalar[ijk + sm1],
                                                 scalar[ijk + sm2]);

                flux[ijk] = (velocity[ijk + sp1] >= 0)*phip + (velocity[ijk + sp1] < 0)*phim;
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}

void hiweno_ninth_order_nonconserv(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 4;
    const ssize_t jmin = 4;
    const ssize_t kmin = 4;

    const ssize_t imax = dims->nlg[0]-5;
    const ssize_t jmax = dims->nlg[1]-5;
    const ssize_t kmax = dims->nlg[2]-5;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sp5 = 5 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;
    const ssize_t sm4 = -4*sp1;

    
    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;

                //Upwind for positive velocity
                const double phip = interp_weno9(scalar[ijk + sm4],
                                                 scalar[ijk + sm3],
                                                 scalar[ijk + sm2],
                                                 scalar[ijk + sm1],
                                                 scalar[ijk],
                                                 scalar[ijk + sp1],
                                                 scalar[ijk + sp2],
                                                 scalar[ijk + sp3],
                                                 scalar[ijk + sp4]);

                // Up wind for negative velocity
                const double phim = interp_weno9(scalar[ijk + sp5],
                                                 scalar[ijk + sp4],
                                                 scalar[ijk + sp3],
                                                 scalar[ijk + sp2],
                                                 scalar[ijk + sp1],
                                                 scalar[ijk],
                                                 scalar[ijk + sm1],
                                                 scalar[ijk + sm2],
                                                 scalar[ijk + sm3]);

                flux[ijk] = (velocity[ijk + sp1] >= 0)*phip + (velocity[ijk + sp1] < 0)*phim;
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}

void hiweno_eleventh_order_nonconserv(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 5;
    const ssize_t jmin = 5;
    const ssize_t kmin = 5;

    const ssize_t imax = dims->nlg[0]-6;
    const ssize_t jmax = dims->nlg[1]-6;
    const ssize_t kmax = dims->nlg[2]-6;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sp4 = 4 * sp1;
    const ssize_t sp5 = 5 * sp1;
    const ssize_t sp6 = 6 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;
    const ssize_t sm3 = -3*sp1;
    const ssize_t sm4 = -4*sp1;
    const ssize_t sm5 = -5*sp1;

    
    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;

                //Upwind for positive velocity
                const double phip = interp_weno11(scalar[ijk + sm5],
                                                  scalar[ijk + sm4],
                                                  scalar[ijk + sm3],
                                                  scalar[ijk + sm2],
                                                  scalar[ijk + sm1],
                                                  scalar[ijk],
                                                  scalar[ijk + sp1],
                                                  scalar[ijk + sp2],
                                                  scalar[ijk + sp3],
                                                  scalar[ijk + sp4],
                                                  scalar[ijk + sp5]);

                // Up wind for negative velocity
                const double phim = interp_weno11(scalar[ijk + sp6],
                                                  scalar[ijk + sp5],
                                                  scalar[ijk + sp4],
                                                  scalar[ijk + sp3],
                                                  scalar[ijk + sp2],
                                                  scalar[ijk + sp1],
                                                  scalar[ijk],
                                                  scalar[ijk + sm1],
                                                  scalar[ijk + sm2],
                                                  scalar[ijk + sm3],
                                                  scalar[ijk + sm4]);

                flux[ijk] = (velocity[ijk + sp1] >= 0)*phip + (velocity[ijk + sp1] < 0)*phim;
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}


void compute_advective_fluxes_a(struct DimStruct *dims, double* restrict rho0, double* rho0_half ,double* restrict velocity, double* restrict scalar,
                                double* restrict flux, int d, int scheme){
    switch(scheme){
        case 1:
            upwind_first_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 2:
            second_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 3:
            weno_third_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 4:
            fourth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 5:
            weno_fifth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 6:
            sixth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 7:
            weno_seventh_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 8:
            eighth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 9:
            weno_ninth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 11:
            weno_eleventh_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        default:
            // Make WENO5 default case. The central schemes may not be necessarily stable, however WENO5 should be.
            weno_fifth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
    };
};

void compute_advective_fluxes_hiweno(struct DimStruct *dims, double* restrict rho0, double* rho0_half ,double* restrict velocity, double* restrict scalar,
                                double* restrict flux, int d, int scheme){
    switch(scheme){
            
        // high order WENO, conservative
        case 3:
            hiweno_third_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 5:
            hiweno_fifth_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 7:
            hiweno_seventh_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 9:
            hiweno_ninth_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 11:
            hiweno_eleventh_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        default:
            hiweno_fifth_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
    };
};


void compute_advective_fluxes_hiweno_nonconserv(struct DimStruct *dims, double* restrict rho0, double* rho0_half ,double* restrict velocity, 
                                                double* restrict scalar, double* restrict flux, int d, int scheme){
    switch(scheme){
            
        // high order WENO, conservative
        case 3:
            hiweno_third_order_nonconserv(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 5:
            hiweno_fifth_order_nonconserv(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 7:
            hiweno_seventh_order_nonconserv(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 9:
            hiweno_ninth_order_nonconserv(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 11:
            hiweno_eleventh_order_nonconserv(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        default:
            hiweno_fifth_order_nonconserv(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
    };
};




void compute_qt_sedimentation_s_source(const struct DimStruct *dims, double *p0_half,  double* rho0_half, double *flux,
                                    double* qt, double* qv, double* T, double* tendency, double (*lam_fp)(double),
                                    double (*L_fp)(double, double), double dx, ssize_t d){

    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;

    const ssize_t imax = dims->nlg[0] - dims->gw;
    const ssize_t jmax = dims->nlg[1] - dims->gw;
    const ssize_t kmax = dims->nlg[2] - dims->gw;

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const double dxi = 1.0/dx;
    const ssize_t stencil[3] = {istride,jstride,1};

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                // Compute Dry air entropy specific entropy
                double pd = pd_c(p0_half[k],qt[ijk],qv[ijk]);
                double sd = sd_c(pd,T[ijk]);

                //Compute water vapor entropy specific entrop
                double pv = pv_c(p0_half[k],qt[ijk],qv[ijk]);
                double sv = sv_c(pv,T[ijk]);

                //Compute water entropy
                double lam = lam_fp(T[ijk]);
                double L = L_fp(T[ijk],lam);
                double sw = sv - (((qt[ijk] - qv[ijk])/qt[ijk])*L/T[ijk]);

                tendency[ijk] -= (sw - sd) / rho0_half[k] * (flux[ijk + stencil[d]] - flux[ijk])*dxi;
            }  // End k loop
        } // End j loop
    } // End i loop
}
