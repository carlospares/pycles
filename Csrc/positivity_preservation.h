#pragma once
#include "grid.h"
#include <math.h>

void positivity_preservation_xu(struct DimStruct *dims, double *u, double *v, double *w, double *ucc, 
                            double *vcc, double *wcc, double *scalars, double *flux, double *tendency, double dt){

    const ssize_t imin = dims->gw - 1;
    const ssize_t jmin = dims->gw - 1;
    const ssize_t kmin = dims->gw - 1;

    const ssize_t imax = dims->nlg[0] - dims->gw;
    const ssize_t jmax = dims->nlg[1] - dims->gw;
    const ssize_t kmax = dims->nlg[2] - dims->gw;

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const double dxi = dims->dxi[0];
    const double dyi = dims->dxi[1];
    const double dzi = dims->dxi[2];

    const ssize_t ip1 = istride;
    const ssize_t ip2 = 2*ip1;
    const ssize_t im1 = -ip1;
    const ssize_t jp1 = jstride;
    const ssize_t jp2 = 2*jp1;
    const ssize_t jm1 = -jp1;
    const ssize_t kp1 = 1;
    const ssize_t kp2 = 2*kp1;
    const ssize_t km1 = -kp1;
    
    double Hkp, Hkm, Hjp, Hjm, Hip, Him;
    double hkp, hkm, hjp, hjm, hip, him;
    double Fkp, Fkm, Fjp, Fjm, Fip, Fim;
    double fp, fm, Gamma, denom, theta;
    double thetakp, thetakm, thetajp, thetajm, thetaip, thetaim;
    double Gkp, Gkm, Gjp, Gjm, Gip, Gim;
    
    double lambda_x = dt*dxi;
    double lambda_y = dt*dyi;
    double lambda_z = dt*dzi;
    
    const ssize_t x_shift = 0;
    const ssize_t y_shift = dims->npg;
    const ssize_t z_shift = dims->npg * 2;
    

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                
                // upwind first order monotone fluxes for f(phi) = {u/v/w}_ctr * phi (u_ctr assumed constant for each cell)
                hip = ucc[ijk]* ((u[ijk + ip1] >= 0)*scalars[ijk] + (u[ijk + ip1] < 0)*scalars[ijk + ip1]);
                hjp = vcc[ijk]* ((v[ijk + jp1] >= 0)*scalars[ijk] + (v[ijk + jp1] < 0)*scalars[ijk + jp1]);
                hkp = wcc[ijk]* ((w[ijk + kp1] >= 0)*scalars[ijk] + (w[ijk + kp1] < 0)*scalars[ijk + kp1]);
                him = ucc[ijk]* ((u[ijk] >= 0)*scalars[ijk + im1] + (u[ijk] < 0)*scalars[ijk]);
                hjm = vcc[ijk]* ((v[ijk] >= 0)*scalars[ijk + jm1] + (v[ijk] < 0)*scalars[ijk]);
                hkm = wcc[ijk]* ((w[ijk] >= 0)*scalars[ijk + km1] + (w[ijk] < 0)*scalars[ijk]);
                
                // high order fluxes for f(phi) = {u/v/w}_ctr * phi
                Hip = ucc[ijk]* flux[x_shift + ijk];
                Hjp = vcc[ijk]* flux[y_shift + ijk];
                Hkp = wcc[ijk]* flux[z_shift + ijk];
                Him = ucc[ijk]* flux[x_shift + ijk + im1];
                Hjm = vcc[ijk]* flux[y_shift + ijk + jm1];
                Hkm = wcc[ijk]* flux[z_shift + ijk + km1];
                
                Fip = -lambda_x*(Hip - hip);
                Fjp = -lambda_y*(Hjp - hjp);
                Fkp = -lambda_z*(Hkp - hkp);
                Fim = lambda_x*(Him - him);
                Fjm = lambda_y*(Hjm - hjm);
                Fkm = lambda_z*(Hkm - hkm);
                
                Gamma = -scalars[ijk] + lambda_x*(hip - him) + lambda_y*(hjp - hjm) + lambda_z*(hkp - hkm);
                denom = Fip*(Fip < 0) + Fim*(Fim < 0) + Fjp*(Fjp < 0) + Fjm*(Fjm < 0) + Fkp*(Fkp < 0) + Fkm*(Fkm < 0);
                theta = fmin(1, fabs(Gamma/denom)); // does this work in C? 
                
                thetaip = (Fip >= 0) ? 1 : theta;
                thetaim = (Fim >= 0) ? 1 : theta;
                thetajp = (Fjp >= 0) ? 1 : theta;
                thetajm = (Fjm >= 0) ? 1 : theta;
                thetakp = (Fkp >= 0) ? 1 : theta;
                thetakm = (Fkm >= 0) ? 1 : theta;
                
                Gip = thetaip*(Hip - hip) + hip;
                Gim = thetaim*(Him - him) + him;
                Gjp = thetajp*(Hjp - hjp) + hjp;
                Gjm = thetajm*(Hjm - hjm) + hjm;
                Gkp = thetakp*(Hkp - hkp) + hkp;
                Gkm = thetakm*(Hkm - hkm) + hkm;
                
                tendency[ijk] += -dxi*(Gip - Gim) - dyi*(Gjp - Gjm) - dzi*(Gkp - Gkm);
                
            } // End k loop
        } // End j loop
    } // End i loop
}