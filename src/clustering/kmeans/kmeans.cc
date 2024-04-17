//  Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not
//  use this file except in compliance with the License. You may obtain a copy
//  of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
//  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
//  License for the specific language governing permissions and limitations
//  under the License
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "clustering/kmeans/kmeans_config.h"
#include "faiss/utils/distances.h"
#include "knowhere/clustering/clustering_factory.h"
#include "knowhere/clustering/clustering_node.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/dataset.h"
#include "knowhere/log.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"
#include "simd/hook.h"

namespace knowhere {

template <typename DataType>
class KmeansClusteringNode : public ClusteringNode {
 public:
    KmeansClusteringNode(const Object& object) {
        static_assert(std::is_same_v<DataType, fp32>, "Kmeans only support float for now");
        pool_ = ThreadPool::GetGlobalBuildThreadPool();
    }

    expected<DataSetPtr>
    Train(const DataSet& dataset, const Config& cfg) override;

    expected<DataSetPtr>
    Assign(const DataSet& dataset, const Config& cfg) override;

    expected<DataSetPtr>
    GetCentroids() override {
        if (!centroids_) {
            LOG_KNOWHERE_ERROR_ << "clustering not trained";
            return expected<DataSetPtr>::Err(Status::clustering_error, "clustering not trained");
        }
        return GenResultDataSet(num_clusters_, dim_, centroids_.get());
    }

    std::string
    Type() const override {
        return knowhere::ClusteringEnum::CLUSTERING_KMEANS;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<KmeansConfig>();
    }

 private:
    void
    computeClosestCentroid(const DataType* vecs, size_t n, const DataType* centroids, uint32_t* closest_centroid,
                           float* closest_centroid_distance);

    void
    initRandom(const DataType* train_data, size_t n_train, uint32_t random_state);

    float
    lloyds_iter(const DataType* train_data, std::vector<std::vector<uint32_t>>& closest_docs,
                uint32_t* closest_centroids, float* closest_centroid_distancessize_t, size_t n_train,
                uint32_t random_state);

    void
    split_clusters(std::vector<int>& hassign, size_t n_train, uint32_t random_state);

    void
    elkan_L2(const DataType* x, const DataType* y, size_t d, size_t nx, size_t ny, uint32_t* ids, float* val);

    void
    exhaustive_L2sqr_blas(const DataType* x, const DataType* y, size_t d, size_t nx, size_t ny, uint32_t* ids,
                          float* val);

 private:
    std::unique_ptr<DataType[]> centroids_ = nullptr;
    std::shared_ptr<ThreadPool> pool_;
    size_t num_clusters_;
    size_t dim_;
};

}  // namespace knowhere

namespace {
#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {
/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */
int
sgemm_(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n, FINTEGER* k, const float* alpha,
       const float* a, FINTEGER* lda, const float* b, FINTEGER* ldb, float* beta, float* c, FINTEGER* ldc);

#ifdef KNOWHERE_WITH_OPENBLAS
void
openblas_set_num_threads(int num_threads);
#endif
}
}  // namespace

namespace knowhere {

template <typename DataType>
expected<DataSetPtr>
KmeansClusteringNode<DataType>::Train(const DataSet& dataset, const Config& cfg) {
    auto kmeans_cfg = static_cast<const KmeansConfig&>(cfg);
    auto num_clusters = kmeans_cfg.num_clusters.value();
    auto random_state = kmeans_cfg.random_state.value();
    auto max_iter = kmeans_cfg.max_iter.value();

    auto rows = dataset.GetRows();
    auto dim = dataset.GetDim();
    auto vecs = dataset.GetTensor();
    centroids_ = std::make_unique<DataType[]>(num_clusters * dim);
    num_clusters_ = num_clusters;
    dim_ = dim;
    knowhere::TimeRecorder build_time("Kmeans train cost", 2);

    initRandom((const DataType*)vecs, rows, random_state);
    LOG_KNOWHERE_INFO_ << "num_clusters: " << num_clusters << " dim: " << dim;

    float old_loss = std::numeric_limits<float>::max();
    std::vector<std::vector<uint32_t>> closest_docs(num_clusters);
    auto centroid_id_mapping = std::make_unique<uint32_t[]>(rows);
    auto closest_centroid_distance = std::make_unique<float[]>(rows);

    for (auto iter = 1; iter <= max_iter; ++iter) {
        auto loss = lloyds_iter((const DataType*)vecs, closest_docs, centroid_id_mapping.get(),
                                closest_centroid_distance.get(), rows, random_state);

        LOG_KNOWHERE_INFO_ << "Iter [" << iter << "/" << max_iter << "], loss: " << loss;
        if ((loss < std::numeric_limits<float>::epsilon()) || ((iter != 1) && ((old_loss - loss) / loss) < 0)) {
            LOG_KNOWHERE_INFO_ << "Residuals unchanged: " << old_loss << " becomes " << loss << ". Early termination.";
            break;
        }
        old_loss = loss;
    }
    build_time.RecordSection("");
    return GenResultDataSet(rows, centroid_id_mapping.release());
}

template <typename DataType>
expected<DataSetPtr>
KmeansClusteringNode<DataType>::Assign(const DataSet& dataset, const Config& cfg) {
    if (!centroids_) {
        LOG_KNOWHERE_ERROR_ << "clustering not trained";
        return expected<DataSetPtr>::Err(Status::clustering_error, "clustering not trained");
    }
    auto rows = dataset.GetRows();
    auto vecs = dataset.GetTensor();
    knowhere::TimeRecorder build_time("Kmeans assign cost", 2);

    auto centroid_id_mapping = std::make_unique<uint32_t[]>(rows);
    auto closest_centroid_distance = std::make_unique<float[]>(rows);
    computeClosestCentroid((const DataType*)vecs, rows, centroids_.get(), centroid_id_mapping.get(),
                           closest_centroid_distance.get());

    build_time.RecordSection("");
    return GenResultDataSet(rows, centroid_id_mapping.release());
}

template <typename DataType>
void
KmeansClusteringNode<DataType>::exhaustive_L2sqr_blas(const DataType* x, const DataType* y, size_t d, size_t nx,
                                                      size_t ny, uint32_t* ids, float* val) {
    static_assert(std::is_same_v<DataType, float>, "sgemm only support float now");
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0) {
        return;
    }
    // let sgemm_ use one thread and parallel outside by threadpool
#ifdef KNOWHERE_WITH_OPENBLAS
    openblas_set_num_threads(1);
#endif

    /* block sizes */
    const size_t bs_x = faiss::distance_compute_blas_query_bs;
    const size_t bs_y = faiss::distance_compute_blas_database_bs;

    std::unique_ptr<float[]> ip_block = std::make_unique<float[]>(bs_x * bs_y);
    std::unique_ptr<float[]> x_norms = std::make_unique<float[]>(nx);
    std::unique_ptr<float[]> y_norms = std::make_unique<float[]>(ny);

    for (size_t i = 0; i < nx; i++) {
        x_norms[i] = faiss::fvec_norm_L2sqr(x + i * d, d);
    }

    for (size_t i = 0; i < ny; i++) {
        y_norms[i] = faiss::fvec_norm_L2sqr(y + i * d, d);
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx) {
            i1 = nx;
        }

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny) {
                j1 = ny;
            }
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose", "Not transpose", &nyi, &nxi, &di, &one, y + j0 * d, &di, x + i0 * d, &di, &zero,
                       ip_block.get(), &nyi);
            }
            for (size_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[j] - 2 * ip;

                    // negative values can occur for identical vectors
                    // due to roundoff errors
                    if (dis < 0)
                        dis = 0;

                    *ip_line = dis;
                    ip_line++;
                    if (j == 0) {
                        ids[i] = j;
                        val[i] = dis;
                    } else if (dis < val[i]) {
                        ids[i] = j;
                        val[i] = dis;
                    }
                }
            }
        }
    }
}

template <typename DataType>
void
KmeansClusteringNode<DataType>::elkan_L2(const DataType* x, const DataType* y, size_t d, size_t nx, size_t ny,
                                         uint32_t* ids, float* val) {
    if (nx == 0 || ny == 0) {
        return;
    }
    const size_t bs_y = 256;
    auto data = std::make_unique<float[]>(bs_y * (bs_y - 1) / 2);

    for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
        size_t j1 = j0 + bs_y;
        if (j1 > ny) {
            j1 = ny;
        }

        auto Y = [&](size_t i, size_t j) -> float& {
            assert(i != j);
            i -= j0, j -= j0;
            return (i > j) ? data[j + i * (i - 1) / 2] : data[i + j * (j - 1) / 2];
        };
        for (size_t i = j0 + 1; i < j1; ++i) {
            const DataType* y_i = y + i * d;
            for (size_t j = j0; j < i; j++) {
                const DataType* y_j = y + j * d;
                Y(i, j) = faiss::fvec_L2sqr(y_i, y_j, d);
            }
        }

        for (size_t i = 0; i < nx; i++) {
            const DataType* x_i = x + i * d;

            int64_t ids_i = j0;
            float val_i = faiss::fvec_L2sqr(x_i, y + j0 * d, d);
            float val_i_time_4 = val_i * 4;
            for (size_t j = j0 + 1; j < j1; j++) {
                if (val_i_time_4 <= Y(ids_i, j)) {
                    continue;
                }
                const DataType* y_j = y + j * d;
                float disij = faiss::fvec_L2sqr(x_i, y_j, d / 2);
                if (disij >= val_i) {
                    continue;
                }
                disij += faiss::fvec_L2sqr(x_i + d / 2, y_j + d / 2, d - d / 2);
                if (disij < val_i) {
                    ids_i = j;
                    val_i = disij;
                    val_i_time_4 = val_i * 4;
                }
            }

            if (j0 == 0 || val[i] > val_i) {
                val[i] = val_i;
                ids[i] = ids_i;
            }
        }
    }
}

template <typename DataType>
void
KmeansClusteringNode<DataType>::initRandom(const DataType* train_data, size_t n_train, uint32_t random_state) {
    std::unordered_set<uint32_t> picked;
    std::mt19937 rng(random_state);
    for (int64_t j = static_cast<int64_t>(n_train) - static_cast<int64_t>(num_clusters_);
         j < static_cast<int64_t>(n_train); ++j) {
        uint32_t tmp = std::uniform_int_distribution<uint32_t>(0, j)(rng);
        if (picked.count(tmp)) {
            tmp = j;
        }
        picked.insert(tmp);
        std::memcpy(centroids_.get() + (j - static_cast<int64_t>(n_train) + static_cast<int64_t>(num_clusters_)) * dim_,
                    train_data + tmp * dim_, dim_ * sizeof(DataType));
    }
}

template <typename DataType>
void
KmeansClusteringNode<DataType>::split_clusters(std::vector<int>& hassign, size_t n_train, uint32_t random_state) {
    /* Take care of void clusters */
    size_t nsplit = 0;
    constexpr float EPS = 1.0 / 1024;
    std::mt19937 mt(random_state);
    for (size_t ci = 0; ci < num_clusters_; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */
            size_t cj;
            for (cj = 0;; cj = (cj + 1) % num_clusters_) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float)(n_train - num_clusters_);
                float r = mt() / float(mt.max());
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }
            std::memcpy(centroids_.get() + ci * dim_, centroids_.get() + cj * dim_, sizeof(DataType) * dim_);

            /* small symmetric pertubation */
            for (size_t j = 0; j < dim_; j++) {
                if (j % 2 == 0) {
                    centroids_[ci * dim_ + j] = centroids_[ci * dim_ + j] * (1 + EPS);
                    centroids_[cj * dim_ + j] = centroids_[cj * dim_ + j] * (1 - EPS);
                } else {
                    centroids_[ci * dim_ + j] = centroids_[ci * dim_ + j] * (1 - EPS);
                    centroids_[cj * dim_ + j] = centroids_[cj * dim_ + j] * (1 + EPS);
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
            LOG_KNOWHERE_INFO_ << ci << " " << cj << " " << hassign[ci] << " " << hassign[cj];
        }
    }
    LOG_KNOWHERE_INFO_ << "there are " << nsplit << " splits";
}

template <typename DataType>
float
KmeansClusteringNode<DataType>::lloyds_iter(const DataType* train_data,
                                            std::vector<std::vector<uint32_t>>& closest_docs,
                                            uint32_t* closest_centroid, float* closest_centroid_distance,
                                            size_t n_train, uint32_t random_state) {
    float losses = 0.0;

    for (size_t c = 0; c < num_clusters_; ++c) {
        closest_docs[c].clear();
    }

    computeClosestCentroid(train_data, n_train, centroids_.get(), closest_centroid, closest_centroid_distance);
    for (size_t i = 0; i < n_train; ++i) {
        closest_docs[closest_centroid[i]].push_back(i);
    }
    std::memset((void*)centroids_.get(), 0x0, num_clusters_ * dim_ * sizeof(DataType));
    std::vector<int> hassign(num_clusters_, 0);

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(num_clusters_);

    // compute the new centroids
    for (size_t i = 0; i < num_clusters_; ++i) {
        futures.emplace_back(pool_->push([&, c = i]() {
            hassign[c] = closest_docs[c].size();
            if (closest_docs[c].empty()) {
                return;
            }
            std::vector<double> centroids_tmp(dim_, 0.0);
            for (size_t ii = 0; ii < closest_docs[c].size(); ii++) {
                if (ii + 1 < closest_docs[c].size()) {
                    auto i1 = closest_docs[c][ii + 1];
                    // this helps a bit
                    _mm_prefetch(train_data + i1 * dim_, _MM_HINT_T0);
                }

                size_t offset = closest_docs[c][ii];
                for (size_t j = 0; j < dim_; ++j) {
                    centroids_tmp[j] += double(train_data[offset * dim_ + j]);
                }
            }
            for (size_t j = 0; j < dim_; ++j) {
                centroids_[c * dim_ + j] = DataType(centroids_tmp[j] / closest_docs[c].size());
            }
        }));
    }
    for (auto& future : futures) {
        future.wait();
    }
    futures.clear();

    for (size_t i = 0; i < n_train; ++i) {
        losses += closest_centroid_distance[i];
    }
    split_clusters(hassign, n_train, random_state);
    return losses;
}

template <typename DataType>
void
KmeansClusteringNode<DataType>::computeClosestCentroid(const DataType* vecs, size_t n, const DataType* centroids,
                                                       uint32_t* closest_centroid, float* closest_centroid_distance) {
    std::vector<folly::Future<folly::Unit>> futures;
    constexpr int block_size = 8192;
    size_t block_num = DIV_ROUND_UP(n, block_size);
    futures.reserve(block_num);
    for (size_t i = 0; i < block_num; ++i) {
        size_t start = i * block_size;
        size_t end = std::min(n, (i + 1) * block_size);
        futures.emplace_back(pool_->push([&, start, end]() {
            if (std::is_same_v<DataType, float>) {
                exhaustive_L2sqr_blas(vecs + start * dim_, centroids, dim_, end - start, num_clusters_,
                                      closest_centroid + start, closest_centroid_distance + start);
            } else {
                elkan_L2(vecs + start * dim_, centroids, dim_, end - start, num_clusters_, closest_centroid + start,
                         closest_centroid_distance + start);
            }
        }));
    }
    for (auto& future : futures) {
        future.wait();
    }
}

// currently only support float
KNOWHERE_CLUSTERING_SIMPLE_REGISTER_GLOBAL(KMEANS, KmeansClusteringNode, fp32);

}  // namespace knowhere
