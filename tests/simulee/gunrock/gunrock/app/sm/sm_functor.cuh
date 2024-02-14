// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.


/**
 * @file sm_functor.cuh
 * @brief Device functions
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/sm/sm_problem.cuh>
#include <gunrock/util/device_intrinsics.cuh>

namespace gunrock {
namespace app {
namespace sm {


/**
 * @brief Structure contains device functions in computing valid degree for each node. 
 *
 * @tparam VertexId    Type used for vertex id (e.g., uint32)
 * @tparam SizeT       Type used for array indexing. (e.g., uint32)
 * @tparam Value       Type used for calculation values (e.g., float)
 * @tparam ProblemData Problem data type which contains data slice for SM problem
 *
 */
template<
  typename VertexId,
  typename SizeT,
  typename Value,
  typename Problem,
  typename _LabelT = VertexId>
struct SMInitFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Vertex mapping condition function. Check if vertex id statisfy label
     * condition based on query graph. This algorithm work for streaming edges
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        bool res = false;
        #pragma unroll 
        for(VertexId i=0; i < d_data_slice -> nodes_query; i++)
        {
            if (d_data_slice->d_data_labels.GetSize()==0 || (__ldg(d_data_slice -> d_data_labels+node) == __ldg(d_data_slice->d_query_labels+i)))
//                    && (d_data_slice -> d_data_ro[node+1] - d_data_slice -> d_data_ro[node] >=
//                    (d_data_slice -> d_query_ro[i+1] - d_data_slice -> d_query_ro[i])))
            {
                res = true;
                break;
            }
        }
        return res;
    }

    /**
     * @brief Vertex mapping apply function. Do nothing.
     * of the node which has the min node id to the max node id.
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        d_data_slice -> d_isValid[node] = true;
    }

    /** 
     * @brief Forward edge on the condition that both source and destination node are in current
     * frontier. Increamentally count degree of each node
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        bool res = false;
//       printf("true label node id:%d, %d, input_item:%d, input_pos:%d\n", s_id, d_id, input_item, input_pos);
        if(d_data_slice->d_isValid[d_id])
        {
            atomicAdd(d_data_slice->d_data_degree+s_id, 1);
            res = true;
        }
        return res;
    }

    /** 
     * @brief Forward Edge Map, compute data graph node ne value: sum of its neighbor 
     * nodes' label values
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
//        printf("(%d,%d)'s degrees: %d, %d\n", s_id, d_id, d_data_slice->d_data_degree[s_id], d_data_slice->d_data_degree[d_id]);
        if(d_data_slice->d_data_labels.GetSize()!=0)
            atomicAdd(d_data_slice->d_data_ne+s_id, d_data_slice->d_data_labels[d_id]);
    }
}; //SMInitFunctor

/**
 * @brief Structure contains device functions in computing valid degree for each node. 
 *
 * @tparam VertexId    Type used for vertex id (e.g., uint32)
 * @tparam SizeT       Type used for array indexing. (e.g., uint32)
 * @tparam Value       Type used for calculation values (e.g., float)
 * @tparam ProblemData Problem data type which contains data slice for SM problem
 *
 */
template<
  typename VertexId,
  typename SizeT,
  typename Value,
  typename Problem,
  typename _LabelT = VertexId>
struct SMFilterFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Vertex mapping condition function. Check if vertex id matches
     * at least one query node's footprint
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        if(node == -1) return false;
        bool res = false;
            if(d_data_slice->d_isValid[node])
            {
                VertexId i;
                #pragma unroll 
                for(i=0; i < d_data_slice -> nodes_query; i++)
                {
                    if ((d_data_slice->d_data_labels.GetSize()==0 && 
                          d_data_slice->d_data_degree[node] >= d_data_slice->d_query_degree[i])
                        || (d_data_slice->d_data_labels.GetSize()!=0
                           && d_data_slice -> d_data_labels[node] == d_data_slice->d_query_labels[i]
                             && ((d_data_slice->d_data_degree[node]==d_data_slice->d_query_degree[i]
                               && d_data_slice->d_data_ne[node]==d_data_slice->d_query_ne[i]) 
                                || (d_data_slice->d_data_degree[node]>d_data_slice->d_query_degree[i]
                                && d_data_slice->d_data_ne[node]>d_data_slice->d_query_ne[i]))))
                    {
                        break;
                    }
                }
//                d_data_slice->filter[0] = false;
                // Current candidate doesn't match any query node's footprint
                if(i==d_data_slice->nodes_query) {
//                    d_data_slice->filter[0] = true;
                    d_data_slice->d_isValid[node] = false;
                    if(d_data_slice->d_data_labels.GetSize()!=0)
                        d_data_slice->d_data_ne[node] = 0;
                    // Update neighbor nodes' NE
                    #pragma unroll
                    for(int j=d_data_slice->d_data_ro[node]; j<d_data_slice->d_data_ro[node+1]; j++)
                    {
                        VertexId neighbor = d_data_slice->d_data_ci[j];
                        if(d_data_slice->d_isValid[neighbor])
                        {
                            atomicSub(d_data_slice->d_data_degree+neighbor, 1);
                            if(d_data_slice->d_data_labels.GetSize()!=0)
                                atomicSub(d_data_slice->d_data_ne+neighbor, d_data_slice->d_data_labels[node]);
                        }
                    }
                }else res = true;
            }
        return res;
    }

    /**
     * @brief Vertex mapping apply function. Do nothing.
     * of the node which has the min node id to the max node id.
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
//        printf("node %d's degree: %d, original degree:%d, ne: %d\n", node, d_data_slice->d_data_degree[node], d_data_slice->d_data_ro[node+1]-d_data_slice->d_data_ro[node], d_data_slice->d_data_ne[node]);
        //TODO: debug mode output
      //  if(d_data_slice->d_data_degree[node]>(d_data_slice->d_data_ro[node+1]-d_data_slice->d_data_ro[node])) printf("Error: computed degree is larger than original degree\n");
    }

}; // SMFilterFunctor


/**
 * @brief Structure contains device functions in distributing work for joining on as many threads
 * as possible.
 *
 * @tparam VertexId    Type used for vertex id (e.g., uint32)
 * @tparam SizeT       Type used for array indexing. (e.g., uint32)
 * @tparam Value       Type used for calculation values (e.g., float)
 * @tparam ProblemData Problem data type which contains data slice for SM problem
 *
 */
template<
  typename VertexId,
  typename SizeT,
  typename Value,
  typename Problem,
  typename _LabelT = VertexId>
struct SMDistributeFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;
    /**
     * @brief Vertex mapping condition function. Check if vertex id matches footprint
     * of the first query vertex.
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        if(node == -1) return false;
        bool res = false;
        VertexId idx = d_data_slice->d_NG[d_data_slice->counter[0]];
        if ((d_data_slice->d_data_labels.GetSize()==0 && 
              d_data_slice->d_data_degree[node] >= d_data_slice->d_query_degree[idx])
            || (d_data_slice->d_data_labels.GetSize()!=0
              && d_data_slice -> d_data_labels[node] == d_data_slice->d_query_labels[idx]
                 && ((d_data_slice->d_data_degree[node]==d_data_slice->d_query_degree[idx]
                   && d_data_slice->d_data_ne[node]==d_data_slice->d_query_ne[idx]) 
                 || (d_data_slice->d_data_degree[node]>d_data_slice->d_query_degree[idx]
                   && d_data_slice->d_data_ne[node]>d_data_slice->d_query_ne[idx]))))
        {
            res = true;
        }
        return res;
    }

    /**
     * @brief Vertex mapping apply function. Do nothing.
     * 
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        d_data_slice->counter[0] = 1;
    }

    /** 
     * @brief Forward edge on the condition that both source and destination node are in current
     * frontier. Increamentally count degree of each node
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
//         printf("s_id:%d, d_id:%d, threadid:%d, input_pos:%d, edge_id:%d \n",s_id, d_id, blockIdx.x * blockDim.x + threadIdx.x, input_pos, edge_id);
        bool res = false;
        SizeT id = d_data_slice->counter[0];
        if(id==d_data_slice->nodes_query) 
            return res;
//       if(input_pos==0 && threadIdx.x + blockIdx.x * blockDim.x ==0) 
//            printf("Level:%d\n", id);
        if(!d_data_slice->d_isValid[d_id] || !d_data_slice->d_isValid[s_id]) return res;
        //TODO: the following two commands only work for triangle like cliques to filter out repeated combinations
        // Try to expand it to 1-look ahead
        if(s_id >= d_id) return res;
        if(id<d_data_slice->nodes_query-1) {
            if(d_id > (d_data_slice->nodes_data-1)) return res; 
            int x;
            for(x=d_data_slice->d_data_ro[d_id]; x<d_data_slice->d_data_ro[d_id+1]; x++)
                if(d_id < d_data_slice->d_data_ci[x]) break;
            if(x==d_data_slice->d_data_ro[d_id+1]) return res;
        }
        //---------------------------------------------------------------------------
         //if(id==2) printf("id:2 first input_pos:%d, partial:%d, s_id:%d, d_id:%d\n", input_pos, d_data_slice->d_partial[input_pos*d_data_slice->nodes_query], s_id, d_id);
        //filter out non-matched nodes
        // check if d_id is same as any node id in current partial result
//        if(input_pos > d_data_slice->edges_data/2) {printf("=======Error: d_partial outof bound======\n"); return res;}
        #pragma unroll
        for(int i=0; i<id-1; i++){
            if(input_pos*d_data_slice->nodes_query+i >= d_data_slice->nodes_query*d_data_slice->edges_data*2/3)
                printf("======here d_partial out of bound=====\n");
            if(d_id == d_data_slice->d_partial[input_pos*d_data_slice->nodes_query+i])
                return res;
        }
        // for level id, get corresponding query node id and check if d_id matches it 
        VertexId idx = d_data_slice->d_NG[id];
        if ((d_data_slice->d_data_labels.GetSize()==0 && 
              d_data_slice->d_data_degree[d_id] >= d_data_slice->d_query_degree[idx])
            || (d_data_slice->d_data_labels.GetSize()!=0
              && d_data_slice -> d_data_labels[d_id] == d_data_slice->d_query_labels[idx]
                 && ((d_data_slice->d_data_degree[d_id]==d_data_slice->d_query_degree[idx]
                   && d_data_slice->d_data_ne[d_id]==d_data_slice->d_query_ne[idx]) 
                 || (d_data_slice->d_data_degree[d_id]>d_data_slice->d_query_degree[idx]
                   && d_data_slice->d_data_ne[d_id]>d_data_slice->d_query_ne[idx]))))
        {
            res = true;
        }
//        if(id==2) printf("second: input_pos:%d, partial:%d, s_id:%d, d_id:%d\n", input_pos, d_data_slice->d_partial[input_pos*d_data_slice->nodes_query], s_id, d_id);
        if(id>1 && res){
            #pragma unroll
            for(int i=d_data_slice->d_NG_ro[id-2]; i<d_data_slice->d_NG_ci.GetSize() && i<d_data_slice->d_NG_ro[id-1]; i++){
                VertexId index = d_data_slice->d_NG_ci[i];
                if(input_pos*d_data_slice->nodes_query+index >= d_data_slice->nodes_query*d_data_slice->edges_data*2/3)
                    printf("=====d_partial out of bound=====\n");
                VertexId dest = d_data_slice->d_partial[input_pos*d_data_slice->nodes_query+index];
                VertexId j;
                // node at position index and at pos id has a non-tree edge
                #pragma unroll
                for(j=d_data_slice->d_data_ro[dest]; j<d_data_slice->d_data_ro[dest+1]; j++) {
                    if(d_id == d_data_slice->d_data_ci[j]) break;
                }
//                printf("s_id:%d, d_id:%d, j:%d, dest:%d, bound:%d\n", s_id, d_id, j, dest, d_data_slice->d_data_ro[dest+1]);
                if(j==d_data_slice->d_data_ro[dest+1]) res = false;
            }
        }

        return res;
    }

    /** 
     * @brief Forward Edge Map, compute data graph node ne value: sum of its neighbor 
     * nodes' label values
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[out] d_data_slice Data slice object.
     * @param[in] edge_id Edge index in the output frontier
     * @param[in] input_item Input Vertex Id
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[out] output_pos Index in the output frontier
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
       if(d_data_slice->counter[0]==1)
            d_data_slice->d_src_node_id[edge_id] = 1;
       else {
            int index = atomicAdd(d_data_slice->num_subs+0, 1);
/*            if(index >= d_data_slice->edges_data*4.5)
                printf("=====d_index and src node exit bound: index:%d, edges:%d===\n", index, d_data_slice->edges_data);*/
            d_data_slice->d_src_node_id[index] = d_id;
            d_data_slice->d_index[index] = input_pos;
        }
    }
}; // SMDistributeFunctor


}  // namespace sm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
