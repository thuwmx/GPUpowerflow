//成功的完成了13659节点和13659节点的计算，结果比较准确

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include "device_functions.h"
//#include "sm_12_atomic_functions.h"
//#include "sm_13_double_functions.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <windows.h>



#include <iostream>
#include <assert.h>
#include <string.h>
#include <cuda_runtime.h>



#include "nicslu.h"

//#include "Utilities.cuh"


#define M_PI 3.14159269589793
//__shared__ double G[13659*13659],B[13659*13659];
__global__ void formYKernel(double *G,double *B,int *lineN,int *from,int *to,double *r,double *x,double *c,double *tr,double *g1,double *b1){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//// printf("%d %d  ",i,j); 
	//// int N=13659;
	//// double trmax;
	//// double gij=r[N*i+j]/(r[N*i+j]*r[N*i+j]+x[N*i+j]*x[N*i+j]);
	//// double bij=-x[N*i+j]/(r[N*i+j]*r[N*i+j]+x[N*i+j]*x[N*i+j]);
	//// if(i!=j){
	////    if (tr[N*i+j]>tr[N*i+j]) trmax=tr[N*i+j];
	//// else trmax=tr[N*i+j];
	////// printf("%f ",trmax);
	////    G[N*i+j]=-trmax*gij;
	////    B[N*i+j]=-trmax*bij;
	//// }
	// int N=13659;
	// if(i<N*N){

	////  if(i==2) printf("%f ",gi);
	//  if((i-i/N)/N==i/N){//只计算上三角
	// double gi=r[i]/(r[i]*r[i]+x[i]*x[i]);
	//    double bi=-x[i]/(r[i]*r[i]+x[i]*x[i]);
	//   if((i-i/N) % N !=0){//不在对角线上
	//   G[i]=-tr[i]*gi;
	//B[i]=-tr[i]*bi;
	//   }
	//   else{//计算对角线元素
	//  double cntg=0,cntb=0;
	//  int j=i/N;//第j个对角元
	//  for(int k=0;k<N;k++){
	//	  double trdirec;
	//	  if (trsign[N*j+k]) trdirec=tr[N*j+k];
	//	  else trdirec=1.0;
	//	  cntg=cntg+trdirec*trdirec*(r[N*j+k]/(r[N*j+k]*r[N*j+k]+x[N*j+k]*x[N*j+k]));
	//      cntb=cntb+trdirec*trdirec*(-x[N*j+k]/(r[N*j+k]*r[N*j+k]+x[N*j+k]*x[N*j+k])+0.5*c[N*j+k]);
	//  }
	//  G[i]=cntg+g1[j];
	//  B[i]=cntb+b1[j];
	//   }
	//  }
	//  else {
	//   G[i]=0;B[i]=0;
	//  }
	// }
	int N=13659,j;
	double cntg,cntb;
	if(i<*lineN){//前lineN个线程用于计算非对角元
		//if(from[i]<to[i]){//只算上三角
		// G[from[i]*N+to[i]]=-tr[i]*(r[i]/(r[i]*r[i]+x[i]*x[i]));
		// B[from[i]*N+to[i]]=-tr[i]*(-x[i]/(r[i]*r[i]+x[i]*x[i]));
		//}else{
		// G[to[i]*N+from[i]]=-tr[i]*(r[i]/(r[i]*r[i]+x[i]*x[i]));
		// B[to[i]*N+from[i]]=-tr[i]*(-x[i]/(r[i]*r[i]+x[i]*x[i]));
		//}
		G[i]=-(r[i]/(r[i]*r[i]+x[i]*x[i]))/tr[i];
		B[i]=-(-x[i]/(r[i]*r[i]+x[i]*x[i]))/tr[i];
	}
	else
		if(i<*lineN+N){//后N个线程用于计算对角元
			j=i-*lineN;
			cntg=0;cntb=0;
			for(int k=0;k<*lineN;k++){
				if(from[k]==j){
					cntg=cntg+(r[k]/(r[k]*r[k]+x[k]*x[k]))/(tr[k]*tr[k]);
					cntb=cntb+(-x[k]/(r[k]*r[k]+x[k]*x[k])+0.5*c[k])/(tr[k]*tr[k]);
				}
				if(to[k]==j){
					cntg=cntg+r[k]/(r[k]*r[k]+x[k]*x[k]);
					cntb=cntb-x[k]/(r[k]*r[k]+x[k]*x[k])+0.5*c[k];
				}
			}
			G[i]=cntg+g1[j];
			B[i]=cntb+b1[j];
		}
}

//double findmax(double *a){
//	double maxa=0.0;
//	for(int i=0;i<n;i++){
//		if (a[i]>maxa) 
//			maxa=a[i];
//	}
//	return(maxa);
//}
//#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//#else
__device__ double MyatomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
			__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
//#endif

__global__ void calc_cntI(double *cntI,int *lineN,int *from,int*to,double *G,double *B,double *V,double *angle,int *NodetoFuncP,int *type)
{   
	//double deltat=0.5;x
	int N=13659;
	int n=N-1;
	int nPV=4091;
	int nfunc=2*n-nPV;
	//double Vj,Vi;
	//	double *fxplus,*fxminus;
	//x[*k]=x[*k]+deltaq;
	long int i = blockIdx.x * blockDim.x + threadIdx.x;

	double deltaP,deltaQ;
	if(i<*lineN){
		if(type[from[i]]!=1){
			deltaP=V[to[i]]*(G[i]*cos(angle[from[i]]-angle[to[i]])+B[i]*sin(angle[from[i]]-angle[to[i]]));
			MyatomicAdd(&cntI[NodetoFuncP[from[i]]],deltaP);
			deltaQ=V[to[i]]*(G[i]*sin(angle[from[i]]-angle[to[i]])-B[i]*cos(angle[from[i]]-angle[to[i]]));
			MyatomicAdd(&cntI[NodetoFuncP[from[i]]+n],deltaQ);
		}
		if(type[to[i]]!=1){
			deltaP=V[from[i]]*(G[i]*cos(angle[to[i]]-angle[from[i]])+B[i]*sin(angle[to[i]]-angle[from[i]]));
			MyatomicAdd(&cntI[NodetoFuncP[to[i]]],deltaP);
			deltaQ=V[from[i]]*(G[i]*sin(angle[to[i]]-angle[from[i]])-B[i]*cos(angle[to[i]]-angle[from[i]]));
			MyatomicAdd(&cntI[NodetoFuncP[to[i]]+n],deltaQ);
		}
	}
	else if(i<*lineN+N){
		int j=i-(*lineN);
		if(type[j]!=1){
			MyatomicAdd(&cntI[NodetoFuncP[j]],V[j]*G[i]);
			MyatomicAdd(&cntI[NodetoFuncP[j]+n],-V[j]*B[i]);
		}
		//if(NodetoFuncP[j]==1) printf("%f %f",V[j]*G[i],cntI[NodetoFuncP[j]]);
	}
}
__global__ void calc_PQ(double *Ptot,double *Qtot,double *V,double *cntI,int *FunctoNode){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int N=13659;
	if(i<N-1)
		Ptot[i]=V[FunctoNode[i]]*cntI[i];
	else if(i<2*(N-1))
		Qtot[i-N+1]=V[FunctoNode[i-N+1]]*cntI[i];
}
__global__ void calc_pf(double *pf,double *Ptot,double *Qtot,int *FunctoNode,int *NodetoFuncP,double *Pg,double *Qg,double *Pl,double *Ql){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int N=13659;
	int n=N-1;
	int nPV=4091;
	int nfunc=2*n-nPV;
	if(i<n){
		int node=FunctoNode[i];
		pf[i]=-(Pg[node]-Pl[node]-Ptot[i]);
	}
	else if(i<nfunc){
		int node=FunctoNode[i];
		pf[i]=-(Qg[node]-Ql[node]-Qtot[NodetoFuncP[node]]);
	}
}
__global__ void changeVAng1(double *V,double *Ang,int *FunctoNode,double *deltat,double *fx1){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int N=13659;
	int n=N-1;
	int nPV=4091;
	int nfunc=2*n-nPV;
	//printf(" %f %f\n",*deltat,fx1[0]);
	if(i<N-1) 
		Ang[FunctoNode[i]]+=(*deltat)*fx1[i];
	else if(i<nfunc)
		V[FunctoNode[i]]+=(*deltat)*fx1[i]*V[FunctoNode[i]];
	//if(i==38)
	//      printf("angle[%d]=%f\n",i,V[nodeV[i]]);

}
__global__ void changeVAng2(double *V,double *Ang,int *FunctoNode,double *deltat,double *fx1,double *fx2){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int N=13659;
	int n=N-1;
	int nPV=4091;
	int nfunc=2*n-nPV;
	if(i<N-1) 
		Ang[FunctoNode[i]]=Ang[FunctoNode[i]]-(*deltat)*fx1[i]+0.5*(*deltat)*(fx2[i]+fx1[i]);
	//       Ang[FunctoNode[i]]=Ang[FunctoNode[i]]+0.5*(*deltat)*(fx2[i]+fx1[i]);
	else if(i<nfunc)
		V[FunctoNode[i]]=V[FunctoNode[i]]-(*deltat)*fx1[i]+(V[FunctoNode[i]]-(*deltat)*fx1[i])*0.5*(*deltat)*(fx2[i]+fx1[i]);
	//	       V[FunctoNode[i]]=V[FunctoNode[i]]+(V[FunctoNode[i]]-(*deltat)*fx1[i])*0.5*(*deltat)*(fx2[i]+fx1[i]);
	//-(*deltat)*fx1[i]

}
int mergeY(int *from,int *to,double *G,double *B,int lineN,int N){
	int i=0;
	while (i<lineN){
		for(int j=0;j<i;j++){
			if(((from[i]==from[j])&&(to[i]==to[j]))||((from[i]==to[j])&&(to[i]==from[j]))){
				G[j]+=G[i];
				B[j]+=B[i];
				for(int k=i;k<lineN-1;k++){
					from[k]=from[k+1];
					to[k]=to[k+1];
					G[k]=G[k+1];
					B[k]=B[k+1];
				}
				for(int k=lineN-1;k<lineN+N-1;k++){
					G[k]=G[k+1];
					B[k]=B[k+1];
				}
				lineN--;
				i--;
			}
		}
		i++;
	}
	return lineN;
}

int formJ(int *Ji,int *Jj,double *J,int *from,int *to,double *G,double *B,double *V,double *ang,double *P,double *Q,int n,int r,int lineN,int *NodetoFuncP,int *NodetoFuncQ,int *FunctoNode,int *type){
	int nnzJ=-1;
	double value;
	for(int i=0;i<lineN;i++){
		if((type[from[i]]!=1)&&(type[to[i]]!=1)){
			//H中两个非零元素
			value=V[from[i]]*(B[i]*cos(ang[from[i]]-ang[to[i]])-G[i]*sin(ang[from[i]]-ang[to[i]]))*V[to[i]];
			if(abs(value)>0.000000001){
				nnzJ++;
				Ji[nnzJ]=NodetoFuncP[from[i]];
				Jj[nnzJ]=NodetoFuncP[to[i]];
				J[nnzJ]=value;
				//if(nnzJ==985)
				//	printf("//");
			}
			value=V[to[i]]*(B[i]*cos(ang[to[i]]-ang[from[i]])-G[i]*sin(ang[to[i]]-ang[from[i]]))*V[from[i]];
			if(abs(value)>0.000000001){
				nnzJ++;
				Ji[nnzJ]=NodetoFuncP[to[i]];
				Jj[nnzJ]=NodetoFuncP[from[i]];
				J[nnzJ]=value;
				//if(nnzJ==985)
				//	printf("//");
			}
			//L中两个非零元素
			if((type[from[i]]==3)&&(type[to[i]]==3)){
				value=V[from[i]]*(B[i]*cos(ang[from[i]]-ang[to[i]])-G[i]*sin(ang[from[i]]-ang[to[i]]))*V[to[i]];
				if(abs(value)>0.000000001){
					nnzJ++;
					Ji[nnzJ]=NodetoFuncQ[from[i]];
					Jj[nnzJ]=NodetoFuncQ[to[i]];
					J[nnzJ]=value;
					//if(nnzJ==985)
					//printf("//");
				}
				value=V[to[i]]*(B[i]*cos(ang[to[i]]-ang[from[i]])-G[i]*sin(ang[to[i]]-ang[from[i]]))*V[from[i]];
				if(abs(value)>0.000000001){
					nnzJ++;
					Ji[nnzJ]=NodetoFuncQ[to[i]];
					Jj[nnzJ]=NodetoFuncQ[from[i]];
					J[nnzJ]=value;
					//if(nnzJ==985)
					//printf("//");
				}
			}
			//N中两个非零元素
			if(type[to[i]]==3){
				value=V[from[i]]*(-G[i]*cos(ang[from[i]]-ang[to[i]])-B[i]*sin(ang[from[i]]-ang[to[i]]))*V[to[i]];
				if(abs(value)>0.000000001){
					nnzJ++;
					Ji[nnzJ]=NodetoFuncP[from[i]];
					Jj[nnzJ]=NodetoFuncQ[to[i]];
					J[nnzJ]=value;
					//if(nnzJ==985)
					//printf("//");
				}
			}
			if(type[from[i]]==3){
				value=V[to[i]]*(-G[i]*cos(ang[to[i]]-ang[from[i]])-B[i]*sin(ang[to[i]]-ang[from[i]]))*V[from[i]];
				if(abs(value)>0.000000001){
					nnzJ++;
					Ji[nnzJ]=NodetoFuncP[to[i]];
					Jj[nnzJ]=NodetoFuncQ[from[i]];
					J[nnzJ]=value;
					//if(nnzJ==985)
					//printf("//");
				}
			}
			//M中两个非零元素
			if(type[from[i]]==3){
				value=V[from[i]]*(G[i]*cos(ang[from[i]]-ang[to[i]])+B[i]*sin(ang[from[i]]-ang[to[i]]))*V[to[i]];
				if(abs(value)>0.000000001){
					nnzJ++;
					Ji[nnzJ]=NodetoFuncQ[from[i]];
					Jj[nnzJ]=NodetoFuncP[to[i]];
					J[nnzJ]=value;
					//if(nnzJ==985)
					//printf("//");
				}
			}
			if(type[to[i]]==3){
				value=V[to[i]]*(G[i]*cos(ang[to[i]]-ang[from[i]])+B[i]*sin(ang[to[i]]-ang[from[i]]))*V[from[i]];
				if(abs(value)>0.000000001){
					nnzJ++;
					Ji[nnzJ]=NodetoFuncQ[to[i]];
					Jj[nnzJ]=NodetoFuncP[from[i]];
					J[nnzJ]=value;
					//if(nnzJ==985)
					//printf("//");
				}
			}
		}
	}
	for(int i=0;i<n;i++){//H对角线元素
		nnzJ++;
		Ji[nnzJ]=i;
		Jj[nnzJ]=i;
		J[nnzJ]=V[FunctoNode[i]]*V[FunctoNode[i]]*B[FunctoNode[i]+lineN]+Q[i];
		//if(nnzJ==985)
		//			printf("//");
	}
	for(int i=0;i<n-r;i++){//L对角线元素
		nnzJ++;
		Ji[nnzJ]=i+n;
		Jj[nnzJ]=i+n;
		J[nnzJ]=V[FunctoNode[i+n]]*V[FunctoNode[i+n]]*B[FunctoNode[i+n]+lineN]-Q[NodetoFuncP[FunctoNode[i+n]]];
		//if(nnzJ==985)
		//			printf("//");
	}
	for(int i=0;i<n-r;i++){//N和M对角线元素
		//if(type[FunctoNode[i]]==3){
		nnzJ++;
		Ji[nnzJ]=NodetoFuncP[FunctoNode[i+n]];
		Jj[nnzJ]=i+n;
		J[nnzJ]=-V[FunctoNode[i+n]]*V[FunctoNode[i+n]]*G[FunctoNode[i+n]+lineN]-P[NodetoFuncP[FunctoNode[i+n]]];
		//if(nnzJ==985)
		//			printf("//");
		nnzJ++;

		Ji[nnzJ]=i+n;
		Jj[nnzJ]=NodetoFuncP[FunctoNode[i+n]];
		J[nnzJ]=V[FunctoNode[i+n]]*V[FunctoNode[i+n]]*G[FunctoNode[i+n]+lineN]-P[NodetoFuncP[FunctoNode[i+n]]];

		//if(nnzJ==985)
		//			printf("//");
	}
	//for(int i=0;i<n+1+lineN;i++)
	//	printf("%d %f %f\n",i,G[i],B[i]);
	//for(int i=0;i<nnzJ;i++)
	//	printf("%d %d %f\n",Ji[i],Jj[i],J[i]);
	return nnzJ+1;
}
void sort(int *col_idx, double *a, int start, int end)
{
	int i, j, it;
	double dt;

	for (i=end-1; i>start; i--)
		for(j=start; j<i; j++)
			if (col_idx[j] > col_idx[j+1]){

				if (a){
					dt=a[j]; 
					a[j]=a[j+1]; 
					a[j+1]=dt;
				}
				it=col_idx[j]; 
				col_idx[j]=col_idx[j+1]; 
				col_idx[j+1]=it;

			}
}
void coo2csr(int n, int nz, double *a, int *i_idx, int *j_idx,
	double *csr_a, int *col_idx, int *row_start)
{
	int i, l;

	for (i=0; i<=n; i++) row_start[i] = 0;

	/* determine row lengths */
	for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;


	for (i=0; i<n; i++) row_start[i+1] += row_start[i];


	/* go through the structure  once more. Fill in output matrix. */
	for (l=0; l<nz; l++){
		i = row_start[i_idx[l]];
		csr_a[i] = a[l];
		col_idx[i] = j_idx[l];
		row_start[i_idx[l]]++;
	}
	/* shift back row_start */
	for (i=n; i>0; i--) row_start[i] = row_start[i-1];

	row_start[0] = 0;

	for (i=0; i<n; i++){
		sort (col_idx, csr_a, row_start[i], row_start[i+1]);
	}
}

int main()
{
//	cudaDeviceReset();
	//	getchar();
	const int N=13659;
	int n=N-1;
	double tLU=0,tanaly=0,tpf=0,tsolve=0,tformY=0,tformJ=0,tchange=0,ttotal=0;
	double t;
	int iteration=100;
	for(int ite=0;ite<iteration;ite++){
    	cudaDeviceReset ( );
		//struct busstation
		//{
		//	double V,ang,Pg,Qg,Pl,Ql;
		//	int type;
		//}bus[N];
		//	int k;
		//	double R[N*N],X[N*N],C[N*N]={0},tr[N*N],shift[N*N];
		double *R = (double*)malloc(5*N*sizeof(double));
		double *X = (double*)malloc(5*N*sizeof(double));
		double *C = (double*)malloc(5*N*sizeof(double));
		double *tr = (double*)malloc(5*N*sizeof(double));
		double *shift = (double*)malloc(5*N*sizeof(double));
		int *from = (int*)malloc(5*N*sizeof(int));
		int *to = (int*)malloc(5*N*sizeof(int));


		double *V = (double*)malloc(N*sizeof(double));
		double *ang = (double*)malloc(N*sizeof(double));
		double *Pg = (double*)malloc(N*sizeof(double));
		double *Qg = (double*)malloc(N*sizeof(double));
		double *Pl = (double*)malloc(N*sizeof(double));
		double *Ql = (double*)malloc(N*sizeof(double));
		double *GG = (double*)malloc(N*sizeof(double));
		double *BB = (double*)malloc(N*sizeof(double));
		int *type = (int*)malloc(N*sizeof(int));
		//double V[N],ang[N],Pg[N],Qg[N],Pl[N],Ql[N],GG[N],BB[N];
		//int type[N];
		long int *node = (long int*)malloc(N*sizeof(long int));
		//	int from[N*N],to[N*N];

		//double inix[2*N];
		int *FunctoNode = (int*)malloc(2*N*sizeof(int));
		int *NodetoFuncP = (int*)malloc(N*sizeof(int));
		int *NodetoFuncQ = (int*)malloc(N*sizeof(int));
		//int FunctoNode[2*N];//nodeAng[i]表示待求ang中第i个的实际节点编号
		//int NodetoFuncP[N],NodetoFuncQ[N];
		//double cstV,cstth;
		//   for(long int i=0;i<N*N;i++){ 
		//       R[i]=1.0e308;
		//	X[i]=1.0e308;
		//	tr[i]=1;
		//}

		FILE *fp;
		if((fp=fopen("net_13659ill_lamda1.002.txt","rt+"))==NULL){
			printf("Cannot open file strike any key exit!");
			getch();
			exit(1);
		}
		int nPV=0,nPQ=0;

		for (int i=0;i<N;i++){
			//fscanf(fp,"%d ",&node);
			fscanf(fp,"%d %lf %lf %lf %lf %lf %lf %lf %lf %d\n",&node[i],&V[i],&ang[i],&Pg[i],&Qg[i],&Pl[i],&Ql[i],&GG[i],&BB[i],&type[i]);
			ang[i]=ang[i]*M_PI/180;	 
			if(type[i]==2){ //PV节点
				//inix[nPQ+nPV]=ang[node-1];
				ang[i]=0;
				FunctoNode[nPQ+nPV]=i;
				NodetoFuncP[i]=nPQ+nPV;
				nPV++;
			}
			if(type[i]==3){ //PQ节点
				//inix[nPQ+N-1]=V[node-1];
				ang[i]=0;
				V[i]=1;
				FunctoNode[nPQ+N-1]=i;
				//inix[nPQ+nPV]=ang[node-1];
				FunctoNode[nPQ+nPV]=i;
				NodetoFuncP[i]=nPQ+nPV;
				NodetoFuncQ[i]=nPQ+N-1;
				nPQ++;
			}
			//if(type[node-1]==1){ //参考节点
			// cstV=V[node-1];
			// cstth=ang[node-1];
			//}
		}
		//for(int i=0;i<N;i++)
		//	printf("%f ",ang[i]);
		int nfunc=2*(N-1)-nPV;
		//printf("%d ",nPV);
		int lineN=0;
		long int fromNode,toNode;
		while (!feof(fp)){
			//fscanf(fp,"%d %d ",&from,&to);
			fscanf(fp,"%d %d %lf %lf %lf %lf %lf\n",&fromNode,&toNode,&R[lineN],&X[lineN],&C[lineN],&tr[lineN],&shift[lineN]);
			for(int i=0;i<N;i++){
				if (node[i]==fromNode) from[lineN]=i;
				if (node[i]==toNode) to[lineN]=i;
			}
			lineN++;
			//R[(to-1)*N+from-1]=R[(from-1)*N+to-1];
			//X[(to-1)*N+from-1]=X[(from-1)*N+to-1];
			//C[(to-1)*N+from-1]=C[(from-1)*N+to-1];
			//trsign[(from-1)*N+to-1]=1;//为标记绕组方向
			//tr[(to-1)*N+from-1]=tr[(from-1)*N+to-1]; 
			// fscanf(fp,"%d",&from);
		}
		fclose(fp);


		double *dev_r,*dev_x,*dev_c,*dev_tr,*dev_b1,*dev_g1,*dev_G,*dev_B;
		int *dev_lineN,*dev_from,*dev_to;
		//double* G = (double *)malloc( N*N*sizeof(double));
		//double* B = (double *)malloc( N*N*sizeof(double));
		/*double G[N*N]={0},B[N*N]={0};*/
		//for(int i=0;i<N*N;i++) G[i]=0;
		/*	clock_t start=clock()*/

		//cudaEvent_t start, stop;
		//   cudaEventCreate(&start);
		//   cudaEventCreate(&stop);
		//   cudaEventRecord(start, 0);



		cudaMalloc((void**)&dev_r,lineN * sizeof(double));

		cudaMemcpy(dev_r,R,lineN * sizeof(double), cudaMemcpyHostToDevice); 

		cudaMalloc((void**)&dev_x,lineN* sizeof(double));
		cudaMemcpy(dev_x,X,lineN* sizeof(double), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_c,lineN * sizeof(double));
		cudaMemcpy(dev_c,C,lineN* sizeof(double), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_tr,lineN * sizeof(double));
		cudaMemcpy(dev_tr,tr,lineN* sizeof(double), cudaMemcpyHostToDevice); 

		//cudaMalloc((void**)&dev_trsign,N*N * sizeof(double));
		//   cudaMemcpy(dev_trsign,trsign,N*N * sizeof(double), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_b1,N * sizeof(double));
		cudaMemcpy(dev_b1,BB,N  * sizeof(double), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_g1,N * sizeof(double));
		cudaMemcpy(dev_g1,GG,N * sizeof(double), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_G,(lineN+N) * sizeof(double));
		cudaMalloc((void**)&dev_B,(lineN+N) * sizeof(double));

		cudaMalloc((void**)&dev_lineN,sizeof(int));
		cudaMemcpy(dev_lineN,&lineN,sizeof(int), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_from,lineN*sizeof(int));
		cudaMemcpy(dev_from,from,lineN*sizeof(int), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_to,lineN*sizeof(int));
		cudaMemcpy(dev_to,to,lineN*sizeof(int), cudaMemcpyHostToDevice); 

		//cudaEventRecord(stop, 0);
		//   cudaEventSynchronize(stop);
		//   double elapsedTime;
		//   cudaEventElapsedTime(&elapsedTime, start, stop);
		//printf("time = %f ",elapsedTime);
		/*	clock_t stop=clock()*/;


		//clock_t stop=clock();
		//double time=(double) (stop-start);
		//printf("time = %f ",time);
		//double G[N*N]={0},B[N*N]={0};
		//cudaMemcpy(G, dev_G,(lineN+N)*sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(B, dev_B,(lineN+N)*sizeof(double), cudaMemcpyDeviceToHost);
		////求得导纳矩阵
		//for(int i=0;i<(lineN+N);i++){
		//	if(i<lineN)
		//	    printf("%d %d %f\n",from[i],to[i],G[i]);
		//	else
		//		printf("%d %d %f\n",i-lineN,i-lineN,G[i]);
		//}

		//printf("%f ",G[36*N+36]);



		//if((fp=fopen("csrLU13659.txt","rt+"))==NULL){
		//     printf("Cannot open file strike any key exit!");
		//     getch();
		//     exit(1);
		//   }

		//int nfunc,nnz;
		//fscanf(fp,"%d %d",&nfunc,&nnz);
		//double* val = (double *)malloc(nnz*sizeof(double));
		//int* colind = (int *)malloc(nnz*sizeof(int));
		//int* rowptr = (int *)malloc((nfunc+1)*sizeof(int));
		//for(int i=0;i<nnz;i++)
		//	fscanf(fp,"%lf",&val[i]);
		//for(int i=0;i<nnz;i++)
		//	fscanf(fp,"%d",&colind[i]);
		//for(int i=0;i<nfunc+1;i++)
		//	fscanf(fp,"%d",&rowptr[i]);
		//fclose(fp);
		////for (int i=0;i<nnzL;i++)
		////		printf("%f\n",valL[i]);
		//double *d_val;
		//int *d_colind,*d_rowptr;
		//cudaMalloc((void**)&d_val,  nnz*sizeof(double));
		//cudaMemcpy(d_val, val, nnz*sizeof(double), cudaMemcpyHostToDevice);
		//cudaMalloc((void**)&d_colind,  nnz*sizeof(int));
		//cudaMemcpy(d_colind, colind, nnz*sizeof(int), cudaMemcpyHostToDevice);
		//cudaMalloc((void**)&d_rowptr,  (nfunc+1)*sizeof(int));
		//cudaMemcpy(d_rowptr, rowptr, (nfunc+1)*sizeof(int), cudaMemcpyHostToDevice);




		double *dev_G2,*dev_B2,*dev_Pg,*dev_Qg,*dev_Pl,*dev_Ql,*dev_V,*dev_angle;
		int *dev_FunctoNode;
		double  *dev_fx1,*dev_fx2;
		double  *dev_pf,*dev_cntI;
		int *dev_NodetoFuncP,*dev_NodetoFuncQ,*dev_type;
		//cudaMalloc((void**)&dev_G2,  N*N*sizeof(double));
		//cudaMemcpy(dev_G2, G,N*N*sizeof(double), cudaMemcpyHostToDevice); 
		//cudaMalloc((void**)&dev_B2,  N*N*sizeof(double));
		//cudaMemcpy(dev_B2, B,N*N*sizeof(double), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_Pg,  N*sizeof(double));
		cudaMemcpy(dev_Pg, Pg,N*sizeof(double), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_Qg,  N*sizeof(double));
		cudaMemcpy(dev_Qg, Qg,N*sizeof(double), cudaMemcpyHostToDevice); 
		cudaMalloc((void**)&dev_Pl,  N*sizeof(double));
		cudaMemcpy(dev_Pl, Pl,N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_Ql,  N*sizeof(double));
		cudaMemcpy(dev_Ql, Ql,N*sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_V,  N*sizeof(double));
		//cudaMemcpy(dev_V, V,N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_FunctoNode,  2*N*sizeof(int));
		cudaMemcpy(dev_FunctoNode, FunctoNode,2*N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_angle,  N*sizeof(double));
		cudaMemcpy(dev_angle, ang,N*sizeof(double), cudaMemcpyHostToDevice);
		//cudaMalloc((void**)&dev_nodeAng,  N*sizeof(int));
		//cudaMemcpy(dev_nodeAng, nodeAng,N*sizeof(int), cudaMemcpyHostToDevice);
		//cudaMalloc((void**)&dev_pfplus,nfunc * sizeof(double));
		//cudaMalloc((void**)&dev_pfminus,nfunc * sizeof(double));
		cudaMalloc((void**)&dev_fx1,nfunc * sizeof(double));
		cudaMalloc((void**)&dev_fx2,nfunc * sizeof(double));


		cudaMalloc((void**)&dev_pf,nfunc * sizeof(double));
		cudaMalloc((void**)&dev_cntI,2*n * sizeof(double));

		cudaMalloc((void**)&dev_NodetoFuncP,N * sizeof(int));
		cudaMemcpy(dev_NodetoFuncP,NodetoFuncP,N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_NodetoFuncQ,N * sizeof(int));
		cudaMemcpy(dev_NodetoFuncQ,NodetoFuncQ,N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_type,N * sizeof(int));
		cudaMemcpy(dev_type,type,N*sizeof(int), cudaMemcpyHostToDevice);

		double *dev_delt;
		cudaMalloc((void**)&dev_delt,  sizeof(double));


		double *fxzeros = (double*)malloc(2*N*sizeof(double));
		for(int i=0;i<2*N;i++){
			fxzeros[i]=0;
		}
		//dim3 threadsPerBlock(256); 
		//dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y); 
		int threads=256;
		int blocksformY=(lineN+N)/threads+1;
		//cudaSetDeviceFlags(cudaDeviceBlockingSync);

		//cudaThreadSynchronize();
		t=0;
		double tmax=50;
		//double delt;
		//cudaEvent_t start,stop;
		double *deltat = (double*)malloc(sizeof(double));
		*deltat=0.01;
		cudaMemcpy(dev_delt, deltat,sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_V, V,N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_angle, ang,N*sizeof(double), cudaMemcpyHostToDevice);

		LARGE_INTEGER t1,t2,tc;
		QueryPerformanceFrequency(&tc);
		QueryPerformanceCounter(&t1);
		LARGE_INTEGER ts,te;
		formYKernel<<<blocksformY,threads>>>(dev_G,dev_B,dev_lineN,dev_from,dev_to,dev_r,dev_x,dev_c,dev_tr,dev_g1,dev_b1);
		cudaThreadSynchronize();
		QueryPerformanceCounter(&t2);
		tformY+=(t2.QuadPart - t1.QuadPart)*1000.0/tc.QuadPart;
		double *G=(double*)malloc((lineN+N)*sizeof(double));
		cudaMemcpy(G, dev_G,(lineN+N)*sizeof(double), cudaMemcpyDeviceToHost);
		double *B=(double*)malloc((lineN+N)*sizeof(double));
		cudaMemcpy(B, dev_B,(lineN+N)*sizeof(double), cudaMemcpyDeviceToHost);

		lineN=mergeY(from,to,G,B,lineN,N);

		cudaMemcpy(dev_G, G,(lineN+N)*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_B, B,(lineN+N)*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_lineN,&lineN,sizeof(int), cudaMemcpyHostToDevice); 
		cudaMemcpy(dev_from,from,lineN*sizeof(int), cudaMemcpyHostToDevice); 
		cudaMemcpy(dev_to,to,lineN*sizeof(int), cudaMemcpyHostToDevice); 
		//for(int i=0;i<lineN+N;i++)
		//	if(i<lineN)
		//		printf("%d %d %f %f\n",from[i],to[i],G[i],B[i]);
		//	else
		//		printf("%d %d %f %f\n",i-lineN,i-lineN,G[i],B[i]);
		double *J=(double*)malloc((N*N)*sizeof(double));
		int *Ji=(int*)malloc((N*N)*sizeof(int));
		int *Jj=(int*)malloc((N*N)*sizeof(int));


		//for(int i=0;i<2*(N-1);i++)
		//	printf("%f ",cntI_all[i]);

		//   printf("formY Time:%f ms\n",(t2.QuadPart - t1.QuadPart)*1000.0/tc.QuadPart);

		free(R);
		free(X);
		free(C);
		free(tr);
		free(shift);




		//// part X://
		//if((fp=fopen("csrJ13659.txt","rt+"))==NULL){
		//     printf("Cannot open file strike any key exit!");
		//     getch();
		//     exit(1);
		//   }
		//int nnz;
		//fscanf(fp,"%d %d",&nfunc,&nnz);



		//for(int i=0;i<nnzJ;i++)
		//	printf("%f ",val[i]);
		//printf("\n");
		//for(int i=0;i<nnzJ;i++)
		//	printf("%d ",colind[i]);
		//printf("\n");
		//for(int i=0;i<nfunc+1;i++)
		//	printf("%d ",rowptr[i]);
		//




		double *angle0 = (double*)malloc(2*sizeof(double));
		double *anglelast = (double*)malloc(sizeof(double));
		*anglelast=0;
		//cublasHandle_t cublasHandle = NULL;
		//cublasCreate(&cublasHandle);
		const double alpha=1.0,beta=0.0;


		int blocksode=nfunc/threads+1;









		//QueryPerformanceCounter(&t22);
		//tanaly+=(t22.QuadPart - t11.QuadPart)*1000.0/tc.QuadPart;
		// 

		double *d_z;  
		cudaMalloc((void**)&d_z, nfunc * sizeof(double));

		double *test=(double*)malloc(nfunc*sizeof(double));
		double *pf=(double*)malloc(nfunc*sizeof(double));
		double *permuPf=(double*)malloc(nfunc*sizeof(double));

		//LARGE_INTEGER t3,t4,tstart,tend;
		//QueryPerformanceFrequency(&tc);
		//   QueryPerformanceCounter(&t3);
		int blocksPQ=(2*n)/threads+1;
		int blockscntI=(N+lineN)/threads+1;
		double *d_Ptot,*d_Qtot;
		cudaMalloc((void**)&d_Ptot, n * sizeof(double));
		cudaMalloc((void**)&d_Qtot, n * sizeof(double));

		double *Ptot=(double*)malloc(N*sizeof(double));
		double *Qtot=(double*)malloc(N*sizeof(double));

		int nnzJ;

		_handle_t solver = NULL;
		_uint_t i;
		_double_t *cfg;
		const _double_t *stat;
		if (__FAIL(NicsLU_Initialize(&solver, &cfg, &stat)))
		{
			printf("Failed to initialize\n");
			system("pause");
			return -1;
		}


		int* rowptrJ = (int *)malloc((nfunc+1)*sizeof(int));


		unsigned int* rowptr = (unsigned int *)malloc((nfunc+1)*sizeof(unsigned int));
		double *d_valL;
		int *d_colindL,*d_rowptrL;

		cudaError_t err;
		err=cudaMalloc((void**)&d_rowptrL,  (nfunc+1)*sizeof(int));
		double *d_valU;
		int *d_colindU,*d_rowptrU;

		err=cudaMalloc((void**)&d_rowptrU,  (nfunc+1)*sizeof(int));

		//float time_elapsed=0;
		//cudaEvent_t start,stop;
		//cudaEventCreate(&start);    //创建Event
  //      cudaEventCreate(&stop);
		double *fx1=(double *)malloc(nfunc*sizeof(double));
		double *fx2=(double *)malloc(nfunc*sizeof(double));
		QueryPerformanceCounter(&ts);
		while (t<tmax){

			cudaMemcpy(dev_cntI, fxzeros,2*n*sizeof(double), cudaMemcpyHostToDevice);

			/*QueryPerformanceCounter(&tstart);*/
			//time_elapsed=0;
			//cudaEventRecord( start,0);

			calc_cntI<<<blockscntI,threads>>>(dev_cntI,dev_lineN,dev_from,dev_to,dev_G,dev_B,dev_V,dev_angle,dev_NodetoFuncP,dev_type);
			cudaThreadSynchronize();

			//cudaMemcpy(test, dev_cntI,nfunc*sizeof(double), cudaMemcpyDeviceToHost);
			//for (int i=0;i<nfunc;i++)
			//	printf("%f ",test[i]);
			calc_PQ<<<blocksPQ,threads>>>(d_Ptot,d_Qtot,dev_V,dev_cntI,dev_FunctoNode);

			cudaThreadSynchronize();
			calc_pf<<<blocksode,threads>>>(dev_pf,d_Ptot,d_Qtot,dev_FunctoNode,dev_NodetoFuncP,dev_Pg,dev_Qg,dev_Pl,dev_Ql);
			//	cudaThreadSynchronize();
			cudaMemcpy(pf, dev_pf,nfunc*sizeof(double), cudaMemcpyDeviceToHost);
			//for (int i=0;i<nfunc;i++)
			//	printf("%f\n",pf[i]);
			//cudaEventRecord( stop,0);    //记录当前时间
			//cudaEventSynchronize(start);    //Waits for an event to complete.
			//cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
			//cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
			//tpf+=time_elapsed;
			cudaMemcpy(Ptot, d_Ptot,n*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Qtot, d_Qtot,n*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(V, dev_V,N*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(ang, dev_angle,N*sizeof(double), cudaMemcpyDeviceToHost);

			QueryPerformanceCounter(&t1);
			nnzJ=formJ(Ji,Jj,J,from,to,G,B,V,ang,Ptot,Qtot,N-1,nPV,lineN,NodetoFuncP,NodetoFuncQ,FunctoNode,type);
			double* val = (double *)malloc(nnzJ*sizeof(double));
		    int* colindJ = (int *)malloc(nnzJ*sizeof(int));
			coo2csr(nfunc,nnzJ,J,Ji,Jj,val,colindJ,rowptrJ);
			QueryPerformanceCounter(&t2);
			tformJ+=(t2.QuadPart - t1.QuadPart)*1000.0/tc.QuadPart;

			unsigned int* colind = (unsigned int *)malloc(nnzJ*sizeof(unsigned int));

			for(int i=0;i<nnzJ;i++)
				colind[i]=(unsigned int)colindJ[i];
			for(int i=0;i<nfunc+1;i++)
				rowptr[i]=(unsigned int)rowptrJ[i];
			//for(int i=0;i<nnzJ;i++)
			//	printf("%f ",val[i]);
			//printf("\n");
			//for(int i=0;i<nnzJ;i++)
			//	printf("%d ",colind[i]);
			//printf("\n");
			//for(int i=0;i<nfunc+1;i++)
			//	printf("%d ",rowptr[i]);
	


			QueryPerformanceFrequency(&tc);
			QueryPerformanceCounter(&t1);


			//    printf("Version %.0lf\nLicense to %.0lf\n", stat[31], stat[29]);


			int cao=NicsLU_Analyze(solver, nfunc, val, colind, rowptr, MATRIX_ROW_REAL, NULL, NULL);
			int cao2=NicsLU_Factorize(solver, val, 1);

			//for(int i=0;i<20;i++)
			//   printf("%f ",stat[i]);
			QueryPerformanceCounter(&t2);

			tLU+=(t2.QuadPart - t1.QuadPart)*1000.0/tc.QuadPart;

            QueryPerformanceCounter(&t1);
		    NicsLU_Solve(solver, pf, fx1);
		    QueryPerformanceCounter(&t2);
		    tsolve+=(t2.QuadPart - t1.QuadPart)*1000.0/tc.QuadPart;
			cudaMemcpy(dev_fx1, fx1,nfunc*sizeof(double), cudaMemcpyHostToDevice);
			//cudaMemcpy(test, dev_fx1,nfunc*sizeof(double), cudaMemcpyDeviceToHost);
			//for (int i=0;i<nfunc;i++)
			//	printf("%f\n",test[i]);
			//cudaEventRecord( start,0);
			changeVAng1<<<blocksode,threads>>>(dev_V,dev_angle,dev_FunctoNode,dev_delt,dev_fx1);
			//cudaEventRecord( stop,0);    //记录当前时间
			//cudaEventSynchronize(start);    //Waits for an event to complete.
			//cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
			//cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
			//tchange+=time_elapsed;
			cudaMemcpy(dev_cntI, fxzeros,nfunc*sizeof(double), cudaMemcpyHostToDevice);

			calc_cntI<<<blockscntI,threads>>>(dev_cntI,dev_lineN,dev_from,dev_to,dev_G,dev_B,dev_V,dev_angle,dev_NodetoFuncP,dev_type);
			cudaThreadSynchronize();
			//cudaMemcpy(test, dev_cntI,nfunc*sizeof(double), cudaMemcpyDeviceToHost);
			//for (int i=0;i<nfunc;i++)
			//	printf("cntI[%d]=%f\n",i,test[i]);
			calc_PQ<<<blocksPQ,threads>>>(d_Ptot,d_Qtot,dev_V,dev_cntI,dev_FunctoNode);

			cudaThreadSynchronize();
			calc_pf<<<blocksode,threads>>>(dev_pf,d_Ptot,d_Qtot,dev_FunctoNode,dev_NodetoFuncP,dev_Pg,dev_Qg,dev_Pl,dev_Ql);
			//	cudaThreadSynchronize();
			cudaMemcpy(pf, dev_pf,nfunc*sizeof(double), cudaMemcpyDeviceToHost);
			QueryPerformanceCounter(&t1);
			NicsLU_Solve(solver, pf, fx2);
			QueryPerformanceCounter(&t2);
			tsolve+=(t2.QuadPart - t1.QuadPart)*1000.0/tc.QuadPart;

			cudaMemcpy(dev_fx2, fx2,nfunc*sizeof(double), cudaMemcpyHostToDevice);
			changeVAng2<<<blocksode,threads>>>(dev_V,dev_angle,dev_FunctoNode,dev_delt,dev_fx1,dev_fx2);

			cudaMemcpy(angle0, dev_angle,2*sizeof(double), cudaMemcpyDeviceToHost);

			t=t+*deltat;
			printf("t= %f  angle[1]=%f\n",t,angle0[FunctoNode[0]]);
			if(abs(angle0[FunctoNode[0]]-*anglelast)<0.0001) break;
			*anglelast=angle0[FunctoNode[0]];
		}
		QueryPerformanceCounter(&te);
		ttotal+=(te.QuadPart - ts.QuadPart)*1000.0/tc.QuadPart;
		//QueryPerformanceCounter(&t4);
		//    printf("solve total Time:%f ms\n",(t4.QuadPart - t3.QuadPart)*1000.0/tc.QuadPart);

		//double time=(double) (stop-start);
		//printf("time = %f ",time);
		free(fx1);
	    free(fx2);
		free(from);
		free(to);
		free(V);
		free(ang);
		free(Pg);
		free(Qg);
		free(Pl);
		free(Ql);
		free(GG);
		free(BB);
		free(type);
		free(node);
		free(NodetoFuncP);
		free(NodetoFuncQ);
		free(FunctoNode);
		free(J);
		free(Ji);
		free(Jj);
		NicsLU_Free(solver);
		cudaFree(dev_fx1);
		cudaFree(dev_fx2);


		cudaFree(dev_r);
		cudaFree(dev_x);
		cudaFree(dev_c);
		cudaFree(dev_tr);
		cudaFree(dev_b1);
		cudaFree(dev_g1);
		//cudaFree(dev_k);
		//cudaFree(dev_delt);
		//cudaFree(fxplus);
		//cudaFree(fxminus);
		cudaFree(d_valL);
		cudaFree(d_rowptrL);
		cudaFree(d_colindL);
		cudaFree(d_valU);
		cudaFree(d_rowptrU);
		cudaFree(d_colindU);
		//cudaFree(dev_G2);
		//cudaFree(dev_B2);
		cudaFree(dev_Pg);
		cudaFree(dev_Qg);
		cudaFree(dev_Pl);
		cudaFree(dev_Ql);
		//cudaFree(dev_pfplus);
		//cudaFree(dev_pfminus);
		cudaFree(dev_V);
		cudaFree(dev_angle);
		cudaFree(dev_FunctoNode);
		cudaFree(dev_type);
		cudaFree(dev_NodetoFuncP);
		cudaFree(dev_NodetoFuncQ);
		cudaFree(dev_type);
		cudaFree(dev_cntI);

		cudaFree(dev_lineN);
		cudaFree(dev_from);
		cudaFree(dev_to);
		cudaFree(dev_G);
		cudaFree(dev_B);
		//cudaEventDestroy(start);    //destory the event
		//cudaEventDestroy(stop);
	}
	printf("iteration times: %f\n",t);
	printf("formY Time:%f ms\n",tformY/iteration);
	printf("formJ Time:%f ms\n",tformJ/iteration);
	printf("J的LU分解 Time:%f ms\n",tLU/iteration);
	printf("solve Time:%f ms\n",tsolve/iteration);
	printf("total time:%f ms\n",ttotal/iteration);


	//free(val);
	//free(rowptr);
	//free(colind);
	//free(valU);
	//free(rowptrU);
	//free(colindU);
	//free(inimax);
	//free(delt);
	//getchar();

	//free(G);
	//free(B);
	return 0;
}