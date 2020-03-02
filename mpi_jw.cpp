#include "common.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)<(b))?(b):(a))
// Apply the force from neighbor to particle
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

typedef struct bin_t {
    vector<particle_t*> *particles;
} bin_t;

MPI_Request request;
int num_bins_per_side;
double bin_size;
int nprocs_per_side;
int nbins_per_procside;
int  jstart;
int  jend;
int  istart;
int  iend;
int  isize;
int  jsize;
vector<vector<bin_t*>*> *bins;
vector<vector<vector<bin_t*>*>*> *collision_bins;  // TODO: Rename to adjacent_bins.
vector<vector<particle_t*>*> *track;
// Make bins .1% larger to account for rounding error in size computation.
// TODO: How does this affect scaling?
#define OVERFILL 1.001

void init_simulation(particle_t* parts, int num_parts, double size,int rank, int num_procs) {
    cout << "Initializing simulation...\n";

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
    if (rank==0){
        cout<<"nprocs_per_side"<<nprocs_per_side<<endl;
	cout<<"nbins_per_procside"<<nbins_per_procside<<endl;
	cout<<"num_bins_per_side"<<num_bins_per_side<<endl;
	cout.flush();
    }
    // Create bins.
    cout << "Creating bins...\n";

    bins = new vector<vector<bin_t*>*>();
    if (rank<nprocs_per_side*nprocs_per_side){
         jstart = (rank/nprocs_per_side)*nbins_per_procside;
	 jend = min(num_bins_per_side,jstart+nbins_per_procside);
	 istart = (rank%nprocs_per_side)*nbins_per_procside;
	 iend = min(istart + nbins_per_procside,num_bins_per_side);
	 isize = iend-istart;
	 jsize = jend-jstart;
        for (int i = 0; i < isize+2; i++) {
            bins->push_back(new vector<bin_t*>());
            for (int j = 0; j < jsize+2; j++) {
                bin_t *bin = new bin_t();
                bin->particles = new vector<particle_t*>();
                (*bins)[i]->push_back(bin);
            }
        }
    
        // Bin particles.
        cout << "Binning particles...\n";
        for (int i = 0; i < num_parts; i++) {
            particle_t *p = parts + i;
    
            int bin_row = p->y / bin_size; // Floor.
            int bin_col = p->x / bin_size; // Floor.
	    if ((bin_row>=istart)&&(bin_row<iend)&&(bin_col>=jstart)&&(bin_col<jend)){
                (*((*bins)[bin_row-istart+1]))[bin_col-jstart+1]->particles->push_back(p);
            }
	}
    
        // Next we will group adjacent bins together that have possibilities of collisions.
        // There will be 9 bins in each group. We will expect there to be on average one
        // collision in each group. This is how we will limit the complexity of the problem.
//        cout << "creating track variable\n";
 //       track=new vector<vector<particle_t*>*>;
//        for (int i =0 ; i< omp_get_num_procs();i++){
 //           track->push_back(new vector<particle_t*>());
 //       }
        cout << "Creating collision bins...\n";
        collision_bins = new vector<vector<vector<bin_t*>*>*>();  // TODO: Having nested pointers is stupid in this case (even the top-most pointer isn't needed).
        for (int i = istart; i < iend; i++) {
	    
            collision_bins->push_back(new vector<vector<bin_t*>*>());
            int iid = i-istart;
            for (int j = jstart; j < jend; j++) {
                (*collision_bins)[iid]->push_back(new vector<bin_t*>());
		int jid = j-jstart;
    
                for (int k = i - 1; k < i + 2; k++) {
                    for (int l = j - 1; l < j + 2; l++) {
                        if (k >= 0 && l >= 0 && k < num_bins_per_side && l < num_bins_per_side) {
                            (*((*collision_bins)[iid]))[jid]->push_back((*((*bins)[k-istart+1]))[l-jstart+1]);  // NOTE: Indexing must be UD/LR otherwise correctness against ref implementation will fail.
                        }
                    }
                }
            }
        }
    }
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs){
//fill nearest bins from other rank
  static int co=0;
  int DONETAG=num_parts+10000;
    int nreceive[num_procs]={ };
    int nreceivereduced[num_procs]={ };
  if (rank<nprocs_per_side*nprocs_per_side){
    //clear the nearest bins out of rank
    for (int i=0;i<isize+2;i+= isize+1){
        for (int j=0;j<jsize+2;j++){
            (*((*bins)[i]))[j]->particles->clear();
        }
    }
    for (int j=0;j<jsize+2;j+=jsize+1){
        for (int i=1;i<isize+1;i++){
            (*((*bins)[i]))[j]->particles->clear();
	}
    }
    // Re-bin particles.

    for (int i = 1; i < isize+1; i++) {
        for (int j = 1; j < jsize+1; j++) {
            vector<particle_t*> *particles = (*((*bins)[i]))[j]->particles;  // TODO: Does saving this cause iteration issues?
            for (int k = 0; k < particles->size(); k++) {
                particle_t *p = (*particles)[k];

                int bin_row = p->y / bin_size; // Floor.
                int bin_col = p->x / bin_size; // Floor.
		int newrank=bin_row/nbins_per_procside+bin_col/nbins_per_procside*nprocs_per_side;

                if (bin_row != i+istart-1 || bin_col != j+jstart-1) {
                    // Remove from current bin.
                    particles->erase(particles->begin()+k);
                    k--;
		    if (newrank==rank){
                        (*((*bins)[bin_row-istart+1]))[bin_col-jstart+1]->particles->push_back(p);
		    }
		    else{
		        MPI_Isend(p,1,PARTICLE,newrank,p-parts,MPI_COMM_WORLD,&request);
                        nreceive[newrank]++;

		    }
		}
            }
        }
    }
  }
    MPI_Allreduce(nreceive,nreceivereduced,num_procs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (rank<nprocs_per_side*nprocs_per_side){
    for (int cc=0;cc<nreceivereduced[rank];cc++){
        MPI_Status status;
        particle_t * particle = new particle_t();
        MPI_Recv(particle,1,PARTICLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
	int loc = status.MPI_TAG;
	parts[loc]=*particle;
        int bin_row = particle->y / bin_size -istart+1; // Floor.
        int bin_col = particle->x / bin_size -jstart+1; // Floor.
        (*((*bins)[bin_row]))[bin_col]->particles->push_back(parts+loc);
    }
  }
    if (rank<nprocs_per_side*nprocs_per_side){
    // send particles at border to nearest rank
    if (iend<num_bins_per_side){
        for (int j=1;j<(jsize+1);j++){
	//send final rows to next rank
	    vector<particle_t*> * particles = (*(*bins)[isize])[j]->particles;
	    for (int l=0;l<particles->size();l++){
	        MPI_Isend((*particles)[l],1,PARTICLE,rank+1,0,MPI_COMM_WORLD,&request);
	    }
	    if ((jend<num_bins_per_side)&&(j==jsize)){
	        for (int l=0;l<particles->size();l++){
	            MPI_Isend((*particles)[l],1,PARTICLE,rank+1+nprocs_per_side,0,MPI_COMM_WORLD,&request);
		}
		MPI_Isend(0,0,PARTICLE,rank+1+nprocs_per_side,DONETAG,MPI_COMM_WORLD,&request);
            }
	    else if ((jstart>0)&&(j==1)){
	        for (int l=0;l<particles->size();l++){
	            MPI_Isend((*particles)[l],1,PARTICLE,rank+1-nprocs_per_side,0,MPI_COMM_WORLD,&request);
		}
		MPI_Isend(0,0,PARTICLE,rank+1-nprocs_per_side,DONETAG,MPI_COMM_WORLD,&request);
	    }
	}
        MPI_Isend(0,0,PARTICLE,rank+1,DONETAG,MPI_COMM_WORLD,&request);
    }

    if (istart>0){
        for (int j=1;j<(jsize+1);j++){
	//send first row to last rank
	    vector<particle_t*> * particles = (*(*bins)[1])[j]->particles;
	    for (int l=0;l<particles->size();l++){
	        MPI_Isend((*particles)[l],1,PARTICLE,rank-1,0,MPI_COMM_WORLD,&request);
	    }
	    if ((jend<num_bins_per_side)&&(j==jsize)){
	        for (int l=0;l<particles->size();l++){
	            MPI_Isend((*particles)[l],1,PARTICLE,rank-1+nprocs_per_side,0,MPI_COMM_WORLD,&request);

		}
		MPI_Isend(0,0,PARTICLE,rank-1+nprocs_per_side,DONETAG,MPI_COMM_WORLD,&request);
            }
	    else if ((jstart>0)&&(j==1)){
	        for (int l=0;l<particles->size();l++){
	            MPI_Isend((*particles)[l],1,PARTICLE,rank-1-nprocs_per_side,0,MPI_COMM_WORLD,&request);
		}
		MPI_Isend(0,0,PARTICLE,rank-1-nprocs_per_side,DONETAG,MPI_COMM_WORLD,&request);
	    }
	}
        MPI_Isend(0,0,PARTICLE,rank-1,DONETAG,MPI_COMM_WORLD,&request);
    }
    
    if (jstart>0){
        //send first column to rank-nprocs_per_side
        for (int i=1;i<(isize+1);i++){
	//send first row to last rank
	    vector<particle_t*> * particles = (*(*bins)[i])[1]->particles;
	    for (int l=0;l<particles->size();l++){
	        MPI_Isend((*particles)[l],1,PARTICLE,rank-nprocs_per_side,0,MPI_COMM_WORLD,&request);
	    }
	}
	MPI_Isend(0,0,PARTICLE,rank-nprocs_per_side,DONETAG,MPI_COMM_WORLD,&request);
    }
    if (jend<num_bins_per_side){
        //send last column to rank+nprocs_per_side
        for (int i=1;i<(isize+1);i++){
	//send first row to last rank
	    vector<particle_t*> * particles = (*(*bins)[i])[jsize]->particles;
	    for (int l=0;l<particles->size();l++){
	        MPI_Isend((*particles)[l],1,PARTICLE,rank+nprocs_per_side,0,MPI_COMM_WORLD,&request);
	    }
	}
	MPI_Isend(0,0,PARTICLE,rank+nprocs_per_side,DONETAG,MPI_COMM_WORLD,&request);
    }
  }
    MPI_Barrier(MPI_COMM_WORLD);
//receive particle data
  if (rank<nprocs_per_side*nprocs_per_side){
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
	    	    if (status.MPI_TAG == DONETAG) break;
	    	    particle_t *p = particle;
                    int bin_row = p->y / bin_size -istart+1; // Floor.
                    int bin_col = p->x / bin_size -jstart+1; // Floor.
                    (*((*bins)[bin_row]))[bin_col]->particles->push_back(p);
		}
	    }
	}
    }
    for (int i = 1; i < (isize+1); i++) {
        for (int j = 1; j < (jsize+1); j++) {
            bin_t *bin = (*((*bins)[i]))[j];
            vector<particle_t*> *particles = bin->particles;
            vector<bin_t*> *adjacent_bins = (*((*collision_bins)[i-1]))[j-1];

            for (int k = 0; k < particles->size(); k++) {
                particle_t *particle = (*particles)[k];
                particle->ax = particle->ay = 0;  // TODO: Handle case where there are no neighbors and setting this would be wrong.
                                                  // NOTE: Reference implementation also makes this error so...

                for (int l = 0; l < adjacent_bins->size(); l++) {
                    vector<particle_t*> *colliding_particles = (*adjacent_bins)[l]->particles;

                    for (int m = 0; m < colliding_particles->size(); m++) {
                        particle_t *colliding_particle = (*colliding_particles)[m];

                        apply_force(*particle, *colliding_particle);
                    }
                }
                move(*particle,size);
            }
        }
    }
/*
    // Move particles.
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
*/
    // Re-bin particles.
    /*
    for (int i = 0; i < num_bins_per_side; i++) {
        for (int j = 0; j < num_bins_per_side; j++) {
            vector<particle_t*> *particles = (*((*bins)[i]))[j]->particles;  // TODO: Does saving this cause iteration issues?
            for (int k = 0; k < particles->size(); k++) {
                particle_t *p = (*particles)[k];

                int bin_row = p->y / bin_size; // Floor.
                int bin_col = p->x / bin_size; // Floor.

                if (bin_row != i || bin_col != j) {
                    // Remove from current bin.
                    particles->erase(particles->begin()+k);
                    k--;
		    int ID=omp_get_thread_num();
                    (*track)[ID]->push_back(p);
                    // Move to correct bin.
                   // (*((*bins)[bin_row]))[bin_col]->particles->push_back(p);
                }
		}
        }
    }
    #pragma omp single
    {
      for (int i=0;i<Nthreads;i++){
        vector<particle_t*> *particles =(*track)[i];
        for (int m=0;m<particles->size();m++){
          particle_t *p =(*particles)[m];
          int bin_row=p->y/bin_size;
          int bin_col=p->x/bin_size;
          (*((*bins)[bin_row]))[bin_col]->particles->push_back(p);
        }
	(*track)[i]->clear();
      }
    }
*/}
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs){
    int DONETAG = num_parts+100;
    if (rank >0 && rank<nprocs_per_side*nprocs_per_side){
        for (int i = 1;i<isize+1;i++){
            for (int j=1;j<jsize+1;j++){
                vector<particle_t*> *particlefrombins=(*(*bins)[i])[j]->particles;
                for (int l=0;l<particlefrombins->size();l++){
                    particle_t* p = (*particlefrombins)[l];
                    MPI_Send(p,1,PARTICLE,0,p-parts,MPI_COMM_WORLD);
        	}
            }
        }
	MPI_Send(0,0,PARTICLE,0,DONETAG,MPI_COMM_WORLD);
    }
    
    if (rank==0){
        int donecc =0;
        MPI_Status status;
	while (donecc < nprocs_per_side*nprocs_per_side-1){
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
}
