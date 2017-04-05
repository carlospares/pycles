#pragma once
#include "grid.h"
#include <math.h>

void positivity_preservation_xu(struct DimStruct *dims, double *ucc, 
                            double *vcc, double *wcc, double *scalars, double *flux, double *tendency, double dt){

	// preserves positivity for advective tendencies
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
    const ssize_t im1 = -ip1;
    const ssize_t jp1 = jstride;
    const ssize_t jm1 = -jp1;
    const ssize_t kp1 = 1;
    const ssize_t km1 = -kp1;
    
    double Hkp, Hkm, Hjp, Hjm, Hip, Him;
    double hkp, hkm, hjp, hjm, hip, him;
    double Fkp, Fkm, Fjp, Fjm, Fip, Fim;
    double fp, fm, Gamma, denom, theta;
    double thetakp, thetakm, thetajp, thetajm, thetaip, thetaim;
    double Gkp, Gkm, Gjp, Gjm, Gip, Gim;
    
    double uip, ui, vjp, vj, wkp, wk;
    
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
                uip = ucc[ijk] + ucc[ijk + ip1];
                ui = ucc[ijk + im1] + ucc[ijk];
                vjp = vcc[ijk] + vcc[ijk + jp1];
                vj = vcc[ijk + jm1] + vcc[ijk];
                wkp = wcc[ijk] + wcc[ijk + ip1];
                wk = wcc[ijk + km1] + wcc[ijk];
                
                hip = ucc[ijk]* ((uip >= 0)*scalars[ijk] + (uip < 0)*scalars[ijk + ip1]);
                hjp = vcc[ijk]* ((vjp >= 0)*scalars[ijk] + (vjp < 0)*scalars[ijk + jp1]);
                hkp = wcc[ijk]* ((wkp >= 0)*scalars[ijk] + (wkp < 0)*scalars[ijk + kp1]);
                him = ucc[ijk]* ((ui >= 0)*scalars[ijk + im1] + (ui < 0)*scalars[ijk]);
                hjm = vcc[ijk]* ((vj >= 0)*scalars[ijk + jm1] + (vj < 0)*scalars[ijk]);
                hkm = wcc[ijk]* ((wk >= 0)*scalars[ijk + km1] + (wk < 0)*scalars[ijk]);
                
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
                theta = fmin(1, fabs(Gamma/denom)); // this works even if denom = 0
                
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
                
                tendency[ijk] += fmax(-dxi*(Gip - Gim) - dyi*(Gjp - Gjm) - dzi*(Gkp - Gkm), -scalars[ijk]/dt);
                
            } // End k loop
        } // End j loop
    } // End i loop
}

void MmP_preservation_xu(struct DimStruct *dims, double *ucc,
                            double *vcc, double *wcc, double *scalars, double *flux, double *tendency, double dt){
	// preserves maximum and minimum principle for the advective tendencies

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
    const ssize_t im1 = -ip1;
    const ssize_t jp1 = jstride;
    const ssize_t jm1 = -jp1;
    const ssize_t kp1 = 1;
    const ssize_t km1 = -kp1;

    double Hkp, Hkm, Hjp, Hjm, Hip, Him;
    double hkp, hkm, hjp, hjm, hip, him;
    double Fkp, Fkm, Fjp, Fjm, Fip, Fim;
    double denomm, thetam, Gammam, denomM, thetaM, GammaM, theta;
    double thetakp, thetakm, thetajp, thetajm, thetaip, thetaim;
    double Gkp, Gkm, Gjp, Gjm, Gip, Gim;

    double uip, ui, vjp, vj, wkp, wk;

    double lambda_x = dt*dxi;
    double lambda_y = dt*dyi;
    double lambda_z = dt*dzi;

    const ssize_t x_shift = 0;
    const ssize_t y_shift = dims->npg;
    const ssize_t z_shift = dims->npg * 2;

    double sm = scalars[imin*istride + jmin*jstride + kmin];
    double sM = scalars[imin*istride + jmin*jstride + kmin];

    // compute extrema to be preserved
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                	sm = fmin(sm, scalars[ijk]);
					sM = fmax(sM, scalars[ijk]);
            }
        }
    }


    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                // upwind first order monotone fluxes for f(phi) = {u/v/w}_ctr * phi (u_ctr assumed constant for each cell)
                uip = ucc[ijk] + ucc[ijk + ip1];
                ui = ucc[ijk + im1] + ucc[ijk];
                vjp = vcc[ijk] + vcc[ijk + jp1];
                vj = vcc[ijk + jm1] + vcc[ijk];
                wkp = wcc[ijk] + wcc[ijk + ip1];
                wk = wcc[ijk + km1] + wcc[ijk];

                hip = ucc[ijk]* ((uip >= 0)*scalars[ijk] + (uip < 0)*scalars[ijk + ip1]);
                hjp = vcc[ijk]* ((vjp >= 0)*scalars[ijk] + (vjp < 0)*scalars[ijk + jp1]);
                hkp = wcc[ijk]* ((wkp >= 0)*scalars[ijk] + (wkp < 0)*scalars[ijk + kp1]);
                him = ucc[ijk]* ((ui >= 0)*scalars[ijk + im1] + (ui < 0)*scalars[ijk]);
                hjm = vcc[ijk]* ((vj >= 0)*scalars[ijk + jm1] + (vj < 0)*scalars[ijk]);
                hkm = wcc[ijk]* ((wk >= 0)*scalars[ijk + km1] + (wk < 0)*scalars[ijk]);

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

                Gammam = sm - scalars[ijk] + lambda_x*(hip - him) + lambda_y*(hjp - hjm) + lambda_z*(hkp - hkm);
                GammaM = sM - scalars[ijk] + lambda_x*(hip - him) + lambda_y*(hjp - hjm) + lambda_z*(hkp - hkm);
                denomm = Fip*(Fip < 0) + Fim*(Fim < 0) + Fjp*(Fjp < 0) + Fjm*(Fjm < 0) + Fkp*(Fkp < 0) + Fkm*(Fkm < 0);
                denomM = Fip*(Fip > 0) + Fim*(Fim > 0) + Fjp*(Fjp > 0) + Fjm*(Fjm > 0) + Fkp*(Fkp > 0) + Fkm*(Fkm > 0);
                thetam = fmin(1, fabs(Gammam/denomm)); // this works even if denom = 0
                thetaM = fmin(1, fabs(GammaM/denomM));
                theta = fmin(thetam,thetaM);

                thetaip = (Fip >= 0) ? 1 : thetam;
                thetaim = (Fim >= 0) ? 1 : thetam;
                thetajp = (Fjp >= 0) ? 1 : thetam;
                thetajm = (Fjm >= 0) ? 1 : thetam;
                thetakp = (Fkp >= 0) ? 1 : thetam;
                thetakm = (Fkm >= 0) ? 1 : thetam;

                Gip = thetaip*(Hip - hip) + hip;
                Gim = thetaim*(Him - him) + him;
                Gjp = thetajp*(Hjp - hjp) + hjp;
                Gjm = thetajm*(Hjm - hjm) + hjm;
                Gkp = thetakp*(Hkp - hkp) + hkp;
                Gkm = thetakm*(Hkm - hkm) + hkm;

//                tendency[ijk] += fmin((sM-scalars[ijk])/dt, fmax(-dxi*(Gip - Gim) - dyi*(Gjp - Gjm) - dzi*(Gkp - Gkm), (sm-scalars[ijk])/dt));
                tendency[ijk] += -dxi*(Gip - Gim) - dyi*(Gjp - Gjm) - dzi*(Gkp - Gkm);

            } // End k loop
        } // End j loop
    } // End i loop
}

