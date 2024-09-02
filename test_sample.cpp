#include "sampler_core.cpp"


using std::cout;
using std::endl;

int main(){
    std::vector<EdgeIDType> indptr{0,2,3,3};
    std::vector<EdgeIDType> indices{1,2, 2};
    std::vector<EdgeIDType> eid{0,1,2};
    std::vector<TimeStampType> ts{5.0f, 3.0f, 1.0f};
    std::vector<int> num_neighbor{1,1};
    auto sampler = ParallelSampler(indptr,indices,eid,ts,1, 1, 2, num_neighbor, true, false, 1, 0.0f);

    std::vector<NodeIDType> root_nodes{0};
    std::vector<TimeStampType> root_ts{10.0f};
    sampler.sample(root_nodes, root_ts);
    return 0;
}