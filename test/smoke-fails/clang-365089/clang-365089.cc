#include <omp.h>

class MAIALbSolver {
    public:
        double* d_distribution = nullptr;
        virtual void propagation_step() = 0;
};

template <int nDist>
class MAIALbSolverDxQy : public MAIALbSolver {
    public:
        virtual void propagation_step() override;
};

template <int nDist>
void MAIALbSolverDxQy<nDist>::propagation_step() {
    #pragma omp target data use_device_ptr(MAIALbSolver::d_distribution)
    {
        const int distStartId = nDist;
        const double* const distributionsStart = &(MAIALbSolver::d_distribution[distStartId]);
    }
};


template class MAIALbSolverDxQy<27>;

int main()
{

  MAIALbSolverDxQy<27> solver;
  solver.propagation_step();
  return 0;
}
