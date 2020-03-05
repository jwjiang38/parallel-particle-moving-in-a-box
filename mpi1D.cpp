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
MPI_Request request1;
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
vector<vector<bin_t*>*> *bins;
vector<bin_t*> * binstosend;
vector<int> * ranktosend;
vector<int> *nearank;
vector<vector<vector<bin_t*>*>*> *collision_bins;  // TODO: Rename to adjacent_bins.
vector<vector<particle_t*>*> *track;
// Make bins .1% larger to account for rounding error in size computation.
// TODO: How does this affect scaling?
#define OVERFILL 1.0

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
    nprocs_per_side = min(num_bins_per_side,num_procs);
    bin_size = (size / num_bins_per_side);
    nbins_per_procside = ceil(float(num_bins_per_side)/float(nprocs_per_side));
    nprocs_per_side = ceil(float(num_bins_per_side)/float(nbins_per_procside));
    nworkrank = nprocs_per_side;
    if (rank==0){
        cout<<"nprocs_per_side"<<nprocs_per_side<<endl;
	cout<<"nbins_per_procside"<<nbins_per_procside<<endl;
	cout<<"num_bins_per_side"<<num_bins_per_side<<endl;
	//cout.flush();
    }

    // Create bins.
   // cout << "Creating bins...\n";

    bins = new vector<vector<bin_t*>*>();
    if (rank<nworkrank){
	 istart = rank*nbins_per_procside;
	 iend = min(istart + nbins_per_procside,num_bins_per_side);
	 isize = iend-istart;
        for (int i = 0; i < isize+2; i++) {
            bins->push_back(new vector<bin_t*>());
            for (int j = 0; j < num_bins_per_side; j++) {
                bin_t *bin = new bin_t();
                bin->particles = new vector<particle_t*>();
                (*bins)[i]->push_back(bin);
            }
        }
    
        // Bin particles.
     //   cout << "Binning particles...\n";
        for (int i = 0; i < num_parts; i++) {
            particle_t *p = parts + i;
    
            int bin_row = p->y / bin_size; // Floor.
            int bin_col = p->x / bin_size; // Floor.
	    if ((bin_row>=istart)&&(bin_row<iend)){
                (*((*bins)[bin_row-istart+1]))[bin_col]->particles->push_back(p);
            }
	}
       /* binstosend = new vector<bin_t*>;
	ranktosend = new vector<int>;
        if (istart>0){
            for (int j=1;j<(jsize+1);j++){
	        binstosend->push_back((*(*bins)[1])[j]);
		ranktosend->push_back(rank-1);
                if ((jend<num_bins_per_side)&&(j==jsize)){
		    binstosend->push_back((*(*bins)[1])[j]);
		    ranktosend->push_back(rank-1+nprocs_per_side);
            	}
                else if ((jstart>0)&&(j==1)){
		    binstosend->push_back((*(*bins)[1])[j]);
		    ranktosend->push_back(rank-1-nprocs_per_side);
                }
            }
        }
        if (iend<num_bins_per_side){
            for (int j=1;j<(jsize+1);j++){
	        binstosend->push_back((*(*bins)[isize])[j]);
		ranktosend->push_back(rank+1);
                if ((jend<num_bins_per_side)&&(j==jsize)){
		    binstosend->push_back((*(*bins)[isize])[j]);
		    ranktosend->push_back(rank+1+nprocs_per_side);
                }
                else if ((jstart>0)&&(j==1)){
		    binstosend->push_back((*(*bins)[isize])[j]);
		    ranktosend->push_back(rank+1-nprocs_per_side);
                }
            }
        }
        if (jstart>0){
            for (int i=1;i<(isize+1);i++){
	        binstosend->push_back((*(*bins)[i])[1]);
		ranktosend->push_back(rank-nprocs_per_side);
            }
	}
        if (jend<num_bins_per_side){
            //send last column to rank+nprocs_per_side
            for (int i=1;i<(isize+1);i++){
	        binstosend->push_back((*(*bins)[i])[jsize]);
		ranktosend->push_back(rank+nprocs_per_side);
            }
	}
    //get near rank list
    nearank = new vector<int>;
    if (istart>0){
        nearank->push_back(rank-1);
	if (jstart>0){
	    nearank->push_back(rank-1-nprocs_per_side);
	}
	if (jend<num_bins_per_side){
	    nearank->push_back(rank-1+nprocs_per_side);
	}
    }
    if (iend<num_bins_per_side){
        nearank->push_back(rank+1);
	if (jstart>0){
	    nearank->push_back(rank+1-nprocs_per_side);
	}
	if (jend<num_bins_per_side){
	    nearank->push_back(rank+1+nprocs_per_side);
	}
    }
    if (jstart>0){
        nearank->push_back(rank-nprocs_per_side);
    }
    if (jend<num_bins_per_side){
        nearank->push_back(rank+nprocs_per_side);
    }*/
        // Next we will group adjacent bins together that have possibilities of collisions.
        // There will be 9 bins in each group. We will expect there to be on average one
        // collision in each group. This is how we will limit the complexity of the problem.
//        cout << "creating track variable\n";
 //       track=new vector<vector<particle_t*>*>;
//        for (int i =0 ; i< omp_get_num_procs();i++){
 //           track->push_back(new vector<particle_t*>());
 //       }
        collision_bins = new vector<vector<vector<bin_t*>*>*>();  // TODO: Having nested pointers is stupid in this case (even the top-most pointer isn't needed).
        for (int i = istart; i < iend; i++) {
	    
            collision_bins->push_back(new vector<vector<bin_t*>*>());
            int iid = i-istart;
            for (int j = 0; j < num_bins_per_side; j++) {
                (*collision_bins)[iid]->push_back(new vector<bin_t*>());
    
                for (int k = i - 1; k < i + 2; k++) {
                    for (int l = j - 1; l < j + 2; l++) {
                        if (k >= 0 && l >= 0 && k < num_bins_per_side && l < num_bins_per_side) {
                            (*((*collision_bins)[iid]))[j]->push_back((*((*bins)[k-istart+1]))[l]);  // NOTE: Indexing must be UD/LR otherwise correctness against ref implementation will fail.
                        }
                    }
                }
            }
        }
    }
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs){
//fill nearest bins from other rank
  int DONETAG=num_parts+10000;
 // int nreceive[num_procs]={ };
 // int nreceivereduced[num_procs]={ };
  if (rank<nworkrank){
    //clear the nearest bins out of rank
        for (int j=0;j<num_bins_per_side;j++){
            (*((*bins)[0]))[j]->particles->clear();
        }
        for (int j=0;j<num_bins_per_side;j++){
            (*((*bins)[isize+1]))[j]->particles->clear();
        }
    // Re-bin particles.

    for (int i = 1; i < isize+1; i++) {
        for (int j = 0; j < num_bins_per_side; j++) {
            vector<particle_t*> *particles = (*((*bins)[i]))[j]->particles;  // TODO: Does saving this cause iteration issues?
            for (int k = 0; k < particles->size(); k++) {
                particle_t *p = (*particles)[k];

                int bin_row = p->y / bin_size; // Floor.
                int bin_col = p->x / bin_size; // Floor.


                if (bin_row != i+istart-1 || bin_col != j) {
                    // Remove from current bin.
                    particles->erase(particles->begin()+k);
                    k--;
		    int newrank=bin_row/nbins_per_procside;
		    if (newrank==rank){
                        (*((*bins)[bin_row-istart+1]))[bin_col]->particles->push_back(p);
		    }
		    else{
		        MPI_Isend(p,1,PARTICLE,newrank,p-parts,MPI_COMM_WORLD,&request);
	//		MPI_Wait(&request,MPI_STATUS_IGNORE);
                //        nreceive[newrank]++;
		    }
		}
            }
        }
    }
 // }
  //  MPI_Allreduce(nreceive,nreceivereduced,num_procs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (rank>0){
      MPI_Isend(0,0,PARTICLE,rank-1,DONETAG,MPI_COMM_WORLD,&request);   
  }
  if (rank<nworkrank-1){
      MPI_Isend(0,0,PARTICLE,rank+1,DONETAG,MPI_COMM_WORLD,&request);   
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
  //receive rebinned particles
  if (rank>0){
      MPI_Status status;
      particle_t* particle = new particle_t();
      while(1){
          MPI_Recv(particle,1,PARTICLE,rank-1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
          int loc = status.MPI_TAG;
          if (loc == DONETAG) break;
          parts[loc] = *particle;
          int bin_row = particle->y / bin_size -istart+1; // Floor.
          int bin_col = particle->x / bin_size; // Floor.

          (*((*bins)[bin_row]))[bin_col]->particles->push_back(parts+loc);

      }

  }
  if (rank<nworkrank-1){
      MPI_Status status;
      particle_t* particle = new particle_t();
      while(1){
          MPI_Recv(particle,1,PARTICLE,rank+1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
          int loc = status.MPI_TAG;
          if (loc == DONETAG) break;
          parts[loc] = *particle;
          int bin_row = particle->y / bin_size -istart+1; // Floor.
          int bin_col = particle->x / bin_size; // Floor.
          (*((*bins)[bin_row]))[bin_col]->particles->push_back(parts+loc);
      }
  }
    //send ghostzone to nearank
    vector<particle_t> * particles = new vector<particle_t>();
    vector<int> *track =  new vector<int>();
    vector<particle_t> * particles1 = new vector<particle_t>();
    vector<int> *track1 =  new vector<int>();
    vector<particle_t> * particles2 = new vector<particle_t>();
    vector<int> *track2 =  new vector<int>();
    vector<particle_t> * particles3 = new vector<particle_t>();
    vector<int> *track3 =  new vector<int>();
    if (rank>0){
        for (int j=0;j<num_bins_per_side;j++){
	    vector<particle_t*> * binp = (*(*bins)[1])[j]->particles;
            for (int k=0;k<binp->size();k++){
	        particles->push_back(*(*binp)[k]);
		track->push_back((*binp)[k]-parts);
	    }
        }
        MPI_Isend(&(*particles)[0],particles->size(),PARTICLE,rank-1,1,MPI_COMM_WORLD,&request);
        MPI_Isend(&(*track)[0],track->size(),MPI_INT,rank-1,1,MPI_COMM_WORLD,&request1);
    }
    if (rank<nworkrank-1){
        for (int j=0;j<num_bins_per_side;j++){
	    vector<particle_t*> * binp = (*(*bins)[isize])[j]->particles;
            for (int k=0;k<binp->size();k++){
	        particles1->push_back(*(*binp)[k]);
		track1->push_back((*binp)[k]-parts);
	    }
        }
        MPI_Isend(&(*particles1)[0],particles1->size(),PARTICLE,rank+1,1,MPI_COMM_WORLD,&request);
        MPI_Isend(&(*track1)[0],track1->size(),MPI_INT,rank+1,1,MPI_COMM_WORLD,&request);
    }
    if (rank>0){
        MPI_Status status;
	particles2->resize(num_parts);
        track2->resize(num_parts);
        MPI_Recv(&(*particles2)[0],num_parts,PARTICLE,rank-1,1,MPI_COMM_WORLD,&status);
        MPI_Recv(&(*track2)[0],num_parts,MPI_INT,rank-1,1,MPI_COMM_WORLD,&status);
	int nparts;
        MPI_Get_count(&status,MPI_INT,&nparts);
	for (int i=0;i<nparts;i++){
	    particle_t *particle = &(*particles2)[i];
	    int loc = (*track2)[i];
            parts[loc]= *particle;
            int bin_row = particle->y / bin_size -istart+1; // Floor.
            int bin_col = particle->x / bin_size; // Floor.
            (*((*bins)[bin_row]))[bin_col]->particles->push_back(parts+loc);
	}
    }
    if (rank<nworkrank-1){
        MPI_Status status;
	particles3->resize(num_parts);
        track3->resize(num_parts);
        MPI_Recv(&(*particles3)[0],num_parts,PARTICLE,rank+1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        MPI_Recv(&(*track3)[0],num_parts,MPI_INT,rank+1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);


	int nparts;
        MPI_Get_count(&status,MPI_INT,&nparts);
	for (int i=0;i<nparts;i++){
	    particle_t* particle =&(*particles3)[i];
	    int loc = (*track3)[i];
            parts[loc]= *particle;
            int bin_row = particle->y / bin_size -istart+1; // Floor.
            int bin_col = particle->x / bin_size; // Floor.
            (*((*bins)[bin_row]))[bin_col]->particles->push_back(parts+loc);
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
        for (int j = 0; j < num_bins_per_side; j++) {
            bin_t *bin = (*((*bins)[i]))[j];
            vector<particle_t*> *particles = bin->particles;
            vector<bin_t*> *adjacent_bins = (*((*collision_bins)[i-1]))[j];
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
            }
        }
    }

    // Move particles.
    for (int i = 1; i < (isize+1); i++) {
        for (int j = 0; j < num_bins_per_side; j++) {
            vector<particle_t*> *particles = (*((*bins)[i]))[j]->particles;
	    for (int l=0;l<particles->size();l++){
	        move(*(*particles)[l],size);
	    }
	}
    }


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
    if (rank==0){
        cout<<"gathering data"<<endl;
	cout.flush();
    }
    int DONETAG = num_parts+100;
    MPI_Request request;
    if (rank >0 && rank<nworkrank){
        for (int i = 1;i<isize+1;i++){
            for (int j=0;j<num_bins_per_side;j++){
                vector<particle_t*> *particlefrombins=(*(*bins)[i])[j]->particles;
                for (int l=0;l<particlefrombins->size();l++){
                    particle_t* p = (*particlefrombins)[l];
                    MPI_Isend(p,1,PARTICLE,0,p-parts,MPI_COMM_WORLD,&request);
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
