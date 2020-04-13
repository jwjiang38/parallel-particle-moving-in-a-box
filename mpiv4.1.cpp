
#include "common.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)<(b))?(b):(a))
// Apply the force from neighbor to particle

int const MAPBIN=6;
typedef struct bin_t {
    particle_t* particles[MAPBIN];
    int nparts=0;
    int nearbins[9];
    int nnbins=0;
} bin_t;

MPI_Request request;
int num_bins_per_side;
double bin_size;
int nprocs_per_side;
int nworkrank;
int nbins_per_procside;
int  jstart;
int  jend;
int  istart;
int  iend;
int  isize;
int  jsize;
bin_t* bins;
int nranks=0;
int nearank[8]={0};
int sendcontrol[16]={0};
//bin_t** collision_bins;  // TODO: Rename to adjacent_bins.
// Make bins .1% larger to account for rounding error in size computation.
// TODO: How does this affect scaling?
#define OVERFILL 1.0

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size,int rank, int num_procs) {
    if (rank==0){
        cout << "Initializing simulation...\n";
	cout.flush();
    }


	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here.

    // We will divide the simulation environment into n 2D bins because we can assume
    // that the simulator is sparse and there will be at most n interactions at each step
    // of the simulation.
    num_bins_per_side = ceil(sqrt(num_parts)); // Ceil.
    //number of procs per side
    nprocs_per_side = min(num_bins_per_side,floor(sqrt(num_procs)));
    bin_size = (size / num_bins_per_side);
    nbins_per_procside = ceil(float(num_bins_per_side)/float(nprocs_per_side));
    nprocs_per_side = ceil(float(num_bins_per_side)/float(nbins_per_procside));
    nworkrank = nprocs_per_side*nprocs_per_side;
    if (rank==0){
        cout<<"nprocs_per_side"<<nprocs_per_side<<endl;
	cout<<"nbins_per_procside"<<nbins_per_procside<<endl;
	cout<<"num_bins_per_side"<<num_bins_per_side<<endl;
	//cout.flush();
    }

    // Create bins.
   // cout << "Creating bins...\n";


    if (rank<nworkrank){

         jstart = (rank/nprocs_per_side)*nbins_per_procside;
	 jend = min(num_bins_per_side,jstart+nbins_per_procside);
	 istart = (rank%nprocs_per_side)*nbins_per_procside;
	 iend = min(istart + nbins_per_procside,num_bins_per_side);
	 isize = iend-istart;
	 jsize = jend-jstart;
         bins = new bin_t[(isize+2)*(jsize+2)]();

        // Bin particles.
        for (int i = 0; i < num_parts; i++) {
            particle_t *p = parts + i;
    
            int bin_row = p->y / bin_size; // Floor.
            int bin_col = p->x / bin_size; // Floor.
	    if ((bin_row>=istart)&&(bin_row<iend)&&(bin_col>=jstart)&&(bin_col<jend)){
	        bin_t* thisbin =&(bins[bin_row-istart+1+(bin_col-jstart+1)*(isize+2)]);
	        thisbin->particles[thisbin->nparts]=p;
	        thisbin->nparts+=1;
            }
	}
        for (int i = istart; i < iend; i++) {
	    
            int iid = i-istart+1;
            for (int j = jstart; j < jend; j++) {
		int jid = j-jstart+1;
                for (int k = i - 1; k < i + 2; k++) {
                    for (int l = j - 1; l < j + 2; l++) {
                        if (k >= 0 && l >= 0 && k < num_bins_per_side && l < num_bins_per_side) {
                            bins[iid+jid*(isize+2)].nearbins[bins[iid+jid*(isize+2)].nnbins]=k-istart+1+(l-jstart+1)*(isize+2);  // NOTE: Indexing must be UD/LR otherwise correctness against ref implementation will fail.
                            bins[iid+jid*(isize+2)].nnbins+=1;  // NOTE: Indexing must be UD/LR otherwise correctness against ref implementation will fail.
                        }
                    }
                }
            }
        }

        if (istart>0){
	    sendcontrol[0]=1;
	    sendcontrol[1]=jsize+1;
        }
        if (iend<num_bins_per_side){
	    sendcontrol[2]=1;
	    sendcontrol[3]=jsize+1;
	}
	if (jstart>0){
	    sendcontrol[4]=1;
	    sendcontrol[5]=isize+1;
	}
        if (jend<num_bins_per_side){
	    sendcontrol[6]=1;
	    sendcontrol[7]=isize+1;
        }
        if (jstart>0 && istart>0){
	    sendcontrol[9]=1;
	}
	if (istart>0 && jend<num_bins_per_side){
	    sendcontrol[11]=1;	
	}
	if (iend<num_bins_per_side && jstart>0){
	    sendcontrol[13]=1;
	}
	if (iend<num_bins_per_side && jend<num_bins_per_side){
	    sendcontrol[15]=1;
	}

    //get near rank list

    if (istart>0){
        nearank[nranks]=(rank-1);
	nranks+=1;
	if (jstart>0){
	    nearank[nranks]=(rank-1-nprocs_per_side);
	    nranks+=1;
	}
	if (jend<num_bins_per_side){
	    nearank[nranks]=(rank-1+nprocs_per_side);
	    nranks+=1;
	}
    }
    if (iend<num_bins_per_side){
        nearank[nranks]=(rank+1);
	nranks+=1;
	if (jstart>0){
	    nearank[nranks]=(rank+1-nprocs_per_side);
	    nranks+=1;
	}
	if (jend<num_bins_per_side){
	    nearank[nranks]=(rank+1+nprocs_per_side);
	    nranks+=1;
	}
    }
    if (jstart>0){
        nearank[nranks]=(rank-nprocs_per_side);
	nranks+=1;
    }
    if (jend<num_bins_per_side){
        nearank[nranks]=(rank+nprocs_per_side);
	nranks+=1;
    }
        // Next we will group adjacent bins together that have possibilities of collisions.
        // There will be 9 bins in each group. We will expect there to be on average one
        // collision in each group. This is how we will limit the complexity of the problem.
//        cout << "creating track variable\n";
 //       track=new vector<vector<particle_t*>*>;
//        for (int i =0 ; i< omp_get_num_procs();i++){
 //           track->push_back(new vector<particle_t*>());
 //       }



    }
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs){
//fill nearest bins from other rank
  int DONETAG=num_parts+10000;
 // int nreceive[num_procs]={ };
 // int nreceivereduced[num_procs]={ };
  if (rank<nworkrank){
    //clear the nearest bins out of rank
    for (int i=0;i<isize+2;i+= isize+1){
        for (int j=0;j<jsize+2;j++){
            bins[i+j*(isize+2)].nparts=0;
        }
    }
    for (int j=0;j<jsize+2;j+=jsize+1){
        for (int i=1;i<isize+1;i++){
            bins[i+j*(isize+2)].nparts=0;
	}
    }
    // Re-bin particles.

    for (int i = 1; i < isize+1; i++) {
        for (int j = 1; j < jsize+1; j++) {
	    bin_t* thisbin= &bins[i+j*(isize+2)];
            for (int k = 0; k < thisbin->nparts; k++) {
                particle_t *p = thisbin->particles[k];

                int bin_row = p->y / bin_size; // Floor.
                int bin_col = p->x / bin_size; // Floor.
		int newrank=bin_row/nbins_per_procside+bin_col/nbins_per_procside*nprocs_per_side;

                if (bin_row != i+istart-1 || bin_col != j+jstart-1) {
                    // Remove from current bin.
		    for (int l=k;l<(thisbin->nparts-1);l++){
		        thisbin->particles[l]=thisbin->particles[l+1];
		    }
		    thisbin->nparts-=1;
                    k--;
		    if (newrank==rank){
		        bin_t* newbin = &bins[bin_row-istart+1+(bin_col-jstart+1)*(isize+2)];
                        newbin->particles[newbin->nparts]=p;
                        newbin->nparts+=1;
		    }
		    else{
		        MPI_Isend(p,1,PARTICLE,newrank,p-parts,MPI_COMM_WORLD,&request);
			MPI_Wait(&request,MPI_STATUS_IGNORE);
                //        nreceive[newrank]++;
		    }
		}
            }
        }
    }
 // }
  //  MPI_Allreduce(nreceive,nreceivereduced,num_procs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  for (int i=0;i<nranks;i++){
      MPI_Isend(0,0,PARTICLE,nearank[i],DONETAG,MPI_COMM_WORLD,&request);   
      MPI_Wait(&request,MPI_STATUS_IGNORE);
  }

//  if (rank<nworkrank){
/*    for (int cc=0;cc<nreceivereduced[rank];cc++){
        MPI_Status status;
        particle_t * particle = new particle_t();
        MPI_Recv(particle,1,PARTICLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
	int loc = status.MPI_TAG;
	parts[loc]=*particle;
        int bin_row = particle->y / bin_size -istart+1; // Floor.
        int bin_col = particle->x / bin_size -jstart+1; // Floor.
        (*((*bins)[bin_row]))[bin_col]->particles->push_back(parts+loc);
    }
  }*/

  for (int i=0;i<nranks;i++){
      int temrank = nearank[i];
      MPI_Status status;
      particle_t * particle = new particle_t();
      while (1){
        MPI_Recv(particle,1,PARTICLE,temrank,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        int loc = status.MPI_TAG;
	if (loc == DONETAG) break;
	parts[loc] = *particle;
        int bin_row = particle->y / bin_size -istart+1; // Floor.
        int bin_col = particle->x / bin_size -jstart+1; // Floor.
	bin_t* thisbin = &bins[bin_row+bin_col*(isize+2)];
        thisbin->particles[thisbin->nparts]=parts+loc;
        thisbin->nparts+=1;
      }
  }
    //send ghostzone to nearank
    for (int j=sendcontrol[0];j<sendcontrol[1];j++){
            bin_t* thisbin = &bins[1+j*(isize+2)];
    	for (int k=0;k<thisbin->nparts;k++){
                MPI_Isend(thisbin->particles[k],1,PARTICLE,rank-1,thisbin->particles[k]-parts,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,MPI_STATUS_IGNORE);
    	}
    }
    for (int j=sendcontrol[2];j<sendcontrol[3];j++){
            bin_t* thisbin = &bins[isize+j*(isize+2)];
    	for (int k=0;k<thisbin->nparts;k++){
                MPI_Isend(thisbin->particles[k],1,PARTICLE,rank+1,thisbin->particles[k]-parts,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,MPI_STATUS_IGNORE);
    	}
    }
    for (int j=sendcontrol[4];j<sendcontrol[5];j++){
            bin_t* thisbin = &bins[j+isize+2];
    	for (int k=0;k<thisbin->nparts;k++){
                MPI_Isend(thisbin->particles[k],1,PARTICLE,rank-nprocs_per_side,thisbin->particles[k]-parts,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,MPI_STATUS_IGNORE);
    	}
    }
    
    for (int j=sendcontrol[6];j<sendcontrol[7];j++){
            bin_t* thisbin = &bins[j+jsize*(isize+2)];
    	for (int k=0;k<thisbin->nparts;k++){
                MPI_Isend(thisbin->particles[k],1,PARTICLE,rank+nprocs_per_side,thisbin->particles[k]-parts,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,MPI_STATUS_IGNORE);
    	}
    }
    for (int j=sendcontrol[8];j<sendcontrol[9];j++){
            bin_t* thisbin = &bins[1+(isize+2)];
    	for (int k=0;k<thisbin->nparts;k++){
                MPI_Isend(thisbin->particles[k],1,PARTICLE,rank-1-nprocs_per_side,thisbin->particles[k]-parts,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,MPI_STATUS_IGNORE);
    	}
    }
    for (int j=sendcontrol[10];j<sendcontrol[11];j++){
            bin_t* thisbin = &bins[1+jsize*(isize+2)];
    	for (int k=0;k<thisbin->nparts;k++){
                MPI_Isend(thisbin->particles[k],1,PARTICLE,rank-1+nprocs_per_side,thisbin->particles[k]-parts,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,MPI_STATUS_IGNORE);
    	}
    }
    for (int j=sendcontrol[12];j<sendcontrol[13];j++){
            bin_t* thisbin = &bins[isize+(isize+2)];
    	for (int k=0;k<thisbin->nparts;k++){
                MPI_Isend(thisbin->particles[k],1,PARTICLE,rank+1-nprocs_per_side,thisbin->particles[k]-parts,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,MPI_STATUS_IGNORE);
    	}
    }
    for (int j=sendcontrol[14];j<sendcontrol[15];j++){
            bin_t* thisbin = &bins[isize+jsize*(isize+2)];
    	for (int k=0;k<thisbin->nparts;k++){
                MPI_Isend(thisbin->particles[k],1,PARTICLE,rank+1+nprocs_per_side,thisbin->particles[k]-parts,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,MPI_STATUS_IGNORE);
    	}
    }
    
    for (int i=0;i<nranks;i++){
        MPI_Isend(0,0,PARTICLE,nearank[i],DONETAG,MPI_COMM_WORLD,&request);   
	MPI_Wait(&request,MPI_STATUS_IGNORE);
    }
    for (int i=0;i<nranks;i++){
        int temrank = nearank[i];
        MPI_Status status;
        particle_t * particle = new particle_t();
        while (1){
          MPI_Recv(particle,1,PARTICLE,temrank,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
          int loc = status.MPI_TAG;
          if (loc == DONETAG) break;
          parts[loc] = *particle;
          int bin_row = particle->y / bin_size -istart+1; // Floor.
          int bin_col = particle->x / bin_size -jstart+1; // Floor.
          bins[bin_row+bin_col*(isize+2)].particles[bins[bin_row+bin_col*(isize+2)].nparts]=parts+loc;
          bins[bin_row+bin_col*(isize+2)].nparts+=1;

        }
    }
  // MPI_Barrier(MPI_COMM_WORLD);
//receive particle data
/*
    int proci = rank%nprocs_per_side;
    int procj = rank/nprocs_per_side;
    for (int i= max(0,proci-1);i<=min(nprocs_per_side-1,proci+1);i++){
        for (int j=max(0,procj-1);j<=min(nprocs_per_side-1,procj+1);j++){
	    int nearank = i+j*nprocs_per_side;
	    if (nearank!=rank){
	        while (1){
	            MPI_Status status;
	            particle_t * particle = new particle_t();
	            MPI_Recv(particle,1,PARTICLE,nearank,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		    int loc = status.MPI_TAG;
	    	    if (loc == DONETAG) break;
	    	    parts[loc] = *particle;
                    int bin_row = particle->y / bin_size -istart+1; // Floor.
                    int bin_col = particle->x / bin_size -jstart+1; // Floor.
                    (*((*bins)[bin_row]))[bin_col]->particles->push_back(parts+loc);
		}
	    }
	}
    }
    */
    for (int i = 1; i < (isize+1); i++) {
        for (int j = 1; j < (jsize+1); j++) {
            bin_t* thisbin = &bins[i+j*(isize+2)];
            particle_t** particles = thisbin->particles;

            for (int k = 0; k < thisbin->nparts; k++) {
                particle_t *particle = particles[k];
                particle->ax = particle->ay = 0;  // TODO: Handle case where there are no neighbors and setting this would be wrong.
                                                  // NOTE: Reference implementation also makes this error so...

                for (int l = 0; l <thisbin->nnbins; l++) {
                    particle_t** colliding_particles = bins[thisbin->nearbins[l]].particles;
                    for (int m = 0; m < bins[thisbin->nearbins[l]].nparts; m++) {
                        particle_t *colliding_particle = colliding_particles[m];

                        apply_force(*particle, *colliding_particle);
                    }
                }
            }
        }
    }

    // Move particles.
    for (int i = 1; i < (isize+1); i++) {
        for (int j = 1; j < (jsize+1); j++) {
            particle_t** particles = bins[i+j*(isize+2)].particles;
	    for (int l=0;l<bins[i+j*(isize+2)].nparts;l++){
	        move(*(particles[l]),size);
	    }
	}
    }


  }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs){
    if (rank==0){
        cout<<"gathering data"<<endl;
	cout.flush();
    }
    int DONETAG = num_parts+100;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Request request;
    if (rank >0 && rank<nworkrank){
        for (int i = 1;i<isize+1;i++){
            for (int j=1;j<jsize+1;j++){
                particle_t** particlefrombins=bins[i+j*(isize+2)].particles;
                for (int l=0;l<bins[i+j*(isize+2)].nparts;l++){
                    particle_t* p = particlefrombins[l];
                    MPI_Isend(p,1,PARTICLE,0,p-parts,MPI_COMM_WORLD,&request);
                    MPI_Wait(&request,MPI_STATUS_IGNORE);
        	}
            }
        }
	MPI_Isend(0,0,PARTICLE,0,DONETAG,MPI_COMM_WORLD,&request);
    }
    if (rank==0){
        int donecc =0;
        MPI_Status status;
	while (donecc < nworkrank-1){
	    particle_t * p0 = new particle_t();
            MPI_Recv(p0,1,PARTICLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            int loc = status.MPI_TAG;
	    if (loc==DONETAG){
	        donecc++;
		continue;
	    }
	    parts[loc]=*p0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
